"""Choosing settings on last season (model/tuning.py): the rules, the comparison on
this season, filing the study, and the dashboard sections (synthetic, no network)."""
import numpy as np
import pandas as pd
import pytest

import config
from fpl_optimiser.model import evaluation, tuning


def _ev(season="2025-26", rounds=(2, 3, 4, 5), noise=0.0, mix=0.7, n=60, seed=0, half_life=8.0, pools=False):
    """A replay whose real points are exactly (1 - mix) x points model + mix x event
    model, plus `noise` on every method's predictions."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in rounds:
        xp, xe = rng.uniform(0, 8, n), rng.uniform(0, 8, n)
        d = pd.DataFrame({"id": np.arange(1, n + 1), "round": r, "fixture_id": 100 * r + np.arange(n),
                          "target_points": (1 - mix) * xp + mix * xe, "minutes": 90, "p_play": 0.9,
                          "trained_to": r - 1, "event_weight": 0.5, "in_pool": np.arange(n) < n // 2})
        d["xp"], d["xp_events"] = xp + rng.normal(0, noise, n), xe + rng.normal(0, noise, n)
        for m in evaluation.METHODS:
            if m not in d:
                d[m] = d["target_points"] + rng.normal(0, 1 + noise, n)
        d["xp_blend"] = 0.5 * d["xp"] + 0.5 * d["xp_events"]
        rows.append(d)
    return evaluation.Evaluation(season=season, rounds=list(rounds), first_week=pd.concat(rows, ignore_index=True),
                                 horizons={}, pools={r: ({1} if pools else set()) for r in rounds},
                                 weights={r: 0.5 for r in rounds}, half_life=half_life)


def test_the_half_life_rule_keeps_the_current_one_unless_clearly_beaten():
    assert tuning.choose_half_life({4.0: 1.960, 8.0: 1.961, 16.0: 1.99}, 8.0) == 8.0      # within TIE
    assert tuning.choose_half_life({4.0: 1.950, 8.0: 1.961, 16.0: 1.99}, 8.0) == 4.0
    assert tuning.choose_half_life({4.0: 1.950, 16.0: 1.99}, 8.0) == 4.0                 # current not tried


def test_the_weight_curve_finds_the_mix():
    curve = tuning.weight_curve(_ev(mix=0.7))
    assert curve.idxmin() == pytest.approx(0.7) and curve.min() == pytest.approx(0.0, abs=1e-9)
    assert curve.loc[0.0] > curve.loc[0.5] > curve.loc[0.7]


def test_two_blends_compared_gameweek_by_gameweek():
    exact, noisy = _ev(), _ev(noise=1.0)
    c = tuning.compare_blends(tuning.blend_rows(exact, 0.7), tuning.blend_rows(noisy, 0.7))
    assert c["rmse"]["new"] == pytest.approx(0.0, abs=1e-9) and c["rmse"]["old"] > 0.5
    assert c["rmse"]["new_better"] == c["rmse"]["gameweeks"] == 4 and c["rmse"]["high"] < 0
    assert c["top11"]["mean_diff"] >= 0 and len(c["gameweeks"]) == 4 and c["n"] == 4 * 60


def test_the_study_chooses_then_replays_in_full(monkeypatch):
    calls = []

    def fake_run(season, table, half_life, event_weight, horizon=config.PLANNING_HORIZON_GW, with_pools=True, **kw):
        calls.append((half_life, horizon, with_pools, event_weight))
        return _ev(season, noise=abs(half_life - 6.0) / 4, half_life=half_life, pools=with_pools)

    from fpl_optimiser.features import multi_season
    monkeypatch.setattr(multi_season, "build_all", lambda seasons=None, half_life=8.0: pd.DataFrame())
    monkeypatch.setattr(evaluation, "run", fake_run)
    study, ev = tuning.run("2025-26", half_lives=(4.0, 6.0, 16.0), before=(8.0, 0.5), log=lambda *_: None)
    w = study["chosen"]["weight"]
    assert study["chosen"]["half_life"] == 6.0 and ev.half_life == 6.0               # the least noise
    assert w == pytest.approx(0.7) and w == pytest.approx(float(study["weight_curve"].idxmin()))
    assert sorted(study["rmse"]) == [4.0, 6.0, 8.0, 16.0] and study["before"] == {"half_life": 8.0, "weight": 0.5}
    assert all(h == 1 and not pools and ew == 0.5 for _, h, pools, ew in calls[:-1])   # quick, at the old weight
    assert calls[-1] == (6.0, config.PLANNING_HORIZON_GW, True, w)                    # full, at the chosen ones
    assert "pool" in study["methods"] and "pool" not in study["half_lives"][4.0]
    assert study["incumbent"] == config.RECENCY_HALF_LIFE_GW                       # the tie protects config's


def test_the_study_is_filed_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "STUDY_DIR", tmp_path)
    assert tuning.load("2025-26") is None
    tuning.save({"season": "2025-26", "version": tuning.VERSION, "chosen": {"half_life": 4.0}})
    assert tuning.load("2025-26")["chosen"]["half_life"] == 4.0
    tuning.save({"season": "2025-26", "version": tuning.VERSION + 1})
    assert tuning.load("2025-26") is None                                              # an older or newer format


def _study():
    ev = _ev(pools=True)
    by_h = {h: tuning.half_life_scores(_ev(noise=h / 16, half_life=h)) for h in (4.0, 8.0)}
    picks = pd.DataFrame([{"method": m, "round": r, "points": p + r, "captain_points": 6.0, "best_captain_points": 8.0,
                           "predicted": 50.0, "cost": 99.0, "captain_id": 1, "subs": 0}
                          for r in (2, 3, 4, 5) for m, p in (("xp_blend", 60), ("xp", 50), ("pos_price_p", 58),
                                                             ("hindsight", 150))])
    season = pd.DataFrame([{"method": m, "net": n, "points": n + 4, "hit_cost": 4, "transfers": 5, "hits": 1,
                            "captain_points": 40.0, "held": n - 10, "transfer_gain": 10.0, "gameweeks": 4}
                           for m, n in (("xp_blend", 230), ("xp", 210), ("pos_price_p", 220))])
    return {"season": "2025-26", "version": tuning.VERSION, "created": "2026-10-26T10:00", "rounds": [2, 3, 4, 5],
            "before": {"half_life": 8.0, "weight": 0.5}, "chosen": {"half_life": 4.0, "weight": 0.7},
            "half_lives": by_h, "rmse": {4.0: 1.9, 8.0: 2.0}, "rule_method": tuning.RULE_METHOD,
            "weight_curve": tuning.weight_curve(ev),
            "methods": tuning.half_life_scores(ev), "rules": "the rules",
            "this_season": {"season": "2026-27", "rounds": [2, 3], "new": {"half_life": 4.0, "weight": 0.7},
                            "old": {"half_life": 8.0, "weight": 0.5},
                            **tuning.compare_blends(tuning.blend_rows(_ev(rounds=(2, 3)), 0.7),
                                                    tuning.blend_rows(_ev(rounds=(2, 3), noise=1.0), 0.5))},
            "decisions": {"picks": picks, "season": season}}


def test_the_last_season_sections_render():
    from fpl_optimiser.app import html, report, tab_accuracy, tab_decisions

    w = report.WeekResults(forecast=None, wildcard=None, last_season=_study())
    r = html.Report("T", "S")
    tab_accuracy.add(r, w, None)
    tab_decisions.add(r, w, None)
    ids = [s.id for s in r.sections]
    assert ids == ["acc-summary", "acc-last-season", "dec-summary", "dec-last-season"]
    assert "acc-weights" in r.figures
    page = html.render(r)
    assert "Last season (2025-26): 4 gameweeks, and where the settings come from" in page
    assert "Checked on this season (2026-27 GW2-GW3)" in page and "Chosen better" in page
    assert "Hindsight best (upper bound)" in page and "A season with transfers, GW2-GW5" in page
    assert tab_accuracy.add(html.Report("T", "S"), report.WeekResults(forecast=None, wildcard=None), None) is None
