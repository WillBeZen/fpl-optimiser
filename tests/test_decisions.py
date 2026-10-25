"""Decision backtest: FPL's scoring of a picked team, weekly picks and a season with
transfers (synthetic, no network)."""
import numpy as np
import pandas as pd
import pytest

import config
from fpl_optimiser.model import evaluation
from fpl_optimiser.optimise import decisions
from tests.test_optimise import _candidates


def _sheet(starters_def=4):
    """A team sheet: ids 1-11 start (GK 1; DEF 2..; MID; FWD), 12-15 the bench in order
    (reserve GK first). Captain 9, vice 10."""
    layout = ["GK"] + ["DEF"] * starters_def + ["MID"] * (8 - starters_def) + ["FWD"] * 2
    bench = [("GK", 12), ("MID", 13), ("DEF", 14), ("FWD", 15)]
    rows = [{"id": i + 1, "position": p, "role": "XI", "bench_order": np.nan} for i, p in enumerate(layout)]
    rows += [{"id": i, "position": p, "role": "bench", "bench_order": k} for k, (p, i) in enumerate(bench)]
    t = pd.DataFrame(rows)
    t["captain"] = t["id"] == 9
    t["vice"] = t["id"] == 10
    return t


def _score(sheet, zero=(), pts=None):
    pts = pts or {i: 2 for i in range(1, 16)}
    mins = {i: (0 if i in zero else 90) for i in range(1, 16)}
    return decisions.realised(sheet, {i: (0 if i in zero else p) for i, p in pts.items()}, mins)


def test_everyone_plays_the_captain_doubles():
    pts = {i: i for i in range(1, 16)}
    res = _score(_sheet(), pts=pts)
    assert res["points"] == sum(range(1, 12)) + 9 and res["subs"] == 0 and res["captain_id"] == 9
    assert res["best_captain_points"] == 11


def test_substitutes_follow_fpls_rules():
    # a defender sits out with four starting: the first bench outfielder who played (a MID) comes on
    res = _score(_sheet(4), zero={2})
    assert res["subs"] == 1 and res["points"] == 11 * 2 + 2
    # with only three defenders starting, a midfielder can't replace one: the bench DEF does
    three = _sheet(3)
    res = _score(three, zero={2, 14})                          # ...unless the bench DEF didn't play either
    assert res["subs"] == 0 and res["points"] == 10 * 2 + 2
    assert _score(three, zero={2})["subs"] == 1
    # a goalkeeper is only replaced by the reserve goalkeeper
    assert _score(_sheet(), zero={1})["subs"] == 1
    assert _score(_sheet(), zero={1, 12})["subs"] == 0


def test_the_vice_captain_doubles_when_the_captain_doesnt_play():
    res = _score(_sheet(), zero={9})
    assert res["captain_id"] == 10 and res["captain_points"] == 2
    res = _score(_sheet(), zero={9, 10})
    assert res["captain_id"] is None and res["captain_points"] == 0


def _evaluation(rounds=(2, 3, 4), injured=None):
    """A replay over a synthetic market: every method predicts each player's xp (the
    reference perfectly, the rest with noise); `injured` scores nothing from GW3 on and
    the replay knows it from then."""
    c = _candidates(n_clubs=12)
    rng = np.random.default_rng(1)
    first, horizons = [], {}
    for r in rounds:
        rows = []
        for g in range(r, r + 3):
            d = c.assign(round=g, fixture_id=1000 * g + c["id"])
            if injured is not None:
                d.loc[(d["id"] == injured) & (g >= 3) & (r >= 3), "xp"] = 0.0
            for m in evaluation.METHODS:
                d[m] = d["xp"] + (0 if m == evaluation.REFERENCE else rng.normal(0, 1.5, len(d)))
            rows.append(d)
        hz = pd.concat(rows, ignore_index=True)
        horizons[r] = hz
        fw = hz[hz["round"] == r].copy()
        out = (injured is not None) & (fw["id"] == injured) & (r >= 3)
        fw["target_points"] = np.where(out, 0, np.round(fw["xp"] + rng.normal(0, 1, len(fw))).clip(0))
        fw["minutes"] = np.where(out, 0, 90)
        first.append(fw)
    ev = evaluation.Evaluation(season="2026-27", rounds=list(rounds), first_week=pd.concat(first, ignore_index=True),
                               horizons=horizons, pools={}, weights={r: 1.0 for r in rounds})
    return ev, c


@pytest.fixture
def market(monkeypatch):
    ev, c = _evaluation(injured=None)
    monkeypatch.setattr(evaluation, "players_at", lambda cur, r: c)
    return ev, c


def test_weekly_picks_are_legal_and_hindsight_is_the_ceiling(market):
    ev, c = market
    picks = decisions.weekly_picks(ev, cur=None, methods=["xp_blend", "season_avg"])
    assert set(picks["method"]) == {"xp_blend", "season_avg", decisions.HINDSIGHT}
    assert (picks["cost"] <= config.BUDGET + 1e-9).all()
    wide = picks.pivot(index="round", columns="method", values="points")
    assert (wide[decisions.HINDSIGHT] >= wide[["xp_blend", "season_avg"]].max(axis=1)).all()
    s = decisions.weekly_summary(picks).set_index("method")
    assert s.at[decisions.HINDSIGHT, "of_hindsight"] == pytest.approx(1.0)


def test_a_season_with_transfers_follows_the_rules(monkeypatch):
    star = _candidates(n_clubs=12).nlargest(1, "xp")["id"].iloc[0]
    ev, c = _evaluation(injured=star)
    monkeypatch.setattr(evaluation, "players_at", lambda cur, r: c)
    s = decisions.simulate(ev, cur=None, method="xp_blend")
    w = s.weeks.set_index("round")
    assert list(w.index) == [2, 3, 4] and w.at[2, "transfers"] == 0      # the first squad is free
    assert (w["net"] == w["points"] - config.TRANSFER_HIT_COST * w["hits"]).all()
    assert (w["bank"] >= 0).all() and w.at[2, "free_after"] == config.FREE_TRANSFERS_PER_GW
    assert w.at[3, "transfers"] >= 1                                     # the injured star is sold
    assert len(s.hold) == 3
    summary = decisions.season_summary({"xp_blend": s}).iloc[0]
    assert summary["transfer_gain"] == pytest.approx(summary["net"] - s.hold["points"].sum())


def test_results_are_cached_until_the_predictions_change(market, tmp_path, monkeypatch):
    ev, c = market
    calls = []
    real = decisions.weekly_picks
    monkeypatch.setattr(decisions, "weekly_picks", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = decisions.run(ev, cur=None, methods=["xp_blend"], cache_dir=tmp_path)
    again = decisions.run(ev, cur=None, methods=["xp_blend"], cache_dir=tmp_path)
    assert not first["cached"] and again["cached"] and len(calls) == 1
    pd.testing.assert_frame_equal(first["picks"], again["picks"])
    ev.first_week["xp_blend"] += 0.5                                      # new predictions: recomputed
    third = decisions.run(ev, cur=None, methods=["xp_blend"], cache_dir=tmp_path)
    assert not third["cached"] and len(calls) == 2


def test_the_decision_tab_renders(market, tmp_path):
    from fpl_optimiser.app import html, report, tab_decisions

    ev, c = market
    dec = decisions.run(ev, cur=None, methods=["xp_blend", "xp", "pos_price_p"], cache_dir=tmp_path)
    dec["actual"] = pd.DataFrame({"round": [2, 3, 4], "points": [60, 50, 70], "hit_cost": [0, 4, 0],
                                  "transfers": [1, 2, 0], "chip": [None, None, "bench_boost"]})
    w = report.WeekResults(forecast=None, wildcard=None, evaluation=ev, decisions=dec)
    r = html.Report("T", "S")
    tab_decisions.add(r, w, None)
    assert [s.id for s in r.sections] == ["dec-summary", "dec-weekly", "dec-season", "dec-captains"]
    assert {"dec-weekly", "dec-season"} <= set(r.figures)
    page = html.render(r)
    assert "Your actual team over GW2-GW4: 176 points after 4 in hits (chips: bench boost GW4)" in page
    assert 'id="tab-decisions"' in page and "Hindsight best (upper bound)" in page
    empty = html.Report("T", "S")
    tab_decisions.add(empty, report.WeekResults(forecast=None, wildcard=None), None)
    assert [s.id for s in empty.sections] == ["dec-summary"]                # --quick: a note
