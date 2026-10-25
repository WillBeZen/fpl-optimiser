"""Model accuracy (the Accuracy tab) — synthetic replays and a filed store, no network."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

import config
from fpl_optimiser.model import accuracy, events, forecast_log
from tests.test_multi_season import two_seasons  # noqa: F401  (pytest fixture)


def _preds():
    """A replay of GW3-5 for three players; player 1 plays twice in GW4."""
    rows = []
    for gw in (3, 4, 5):
        for pid, pred, act, mins in ((1, 6.0, 8, 90), (2, 3.0, 1, 90), (3, 2.5, 0, 0)):
            rows.append({"id": pid, "round": gw, "xp": pred - 0.5, "xp_events": pred + 0.5, "xp_blend": pred,
                         "target_points": act, "minutes": mins, "p_play": 0.9, "trained_to": gw - 1,
                         "half_life": config.RECENCY_HALF_LIFE_GW, "events_half_life": config.RECENCY_HALF_LIFE_GW,
                         **{s: 0.0 for s in events.SOURCES}, "Goals": pred / 2})
    rows.append({**rows[3], "xp_blend": 2.0, "xp": 1.5, "xp_events": 2.5, "target_points": 3, "Goals": 1.0})
    return pd.DataFrame(rows)


def test_window_is_the_last_finished_gameweeks():
    gws = pd.DataFrame({"id": range(1, 10), "finished": [True] * 7 + [False] * 2})
    assert accuracy.window(gws, 5) == [3, 4, 5, 6, 7]
    assert accuracy.window(gws, 10) == list(range(1, 8))


def test_player_gameweeks_add_up_doubles_and_flag_missing_parts():
    pg = accuracy.player_gameweeks(_preds(), [3, 4, 5]).set_index(["id", "round"])
    assert pg.loc[(1, 4), "predicted"] == 8.0 and pg.loc[(1, 4), "actual"] == 11      # 6 + 2, 8 + 3
    assert pg.loc[(1, 4), "fixtures"] == 2 and pg.loc[(1, 4), "error"] == -3.0
    assert len(accuracy.player_gameweeks(_preds(), [3], ids=[2])) == 1
    gap = _preds()
    gap.loc[0, "xp_blend"] = np.nan                       # one fixture without a prediction
    assert np.isnan(accuracy.player_gameweeks(gap, [3]).set_index("id").at[1, "predicted"])


def test_summary_numbers():
    pg = accuracy.player_gameweeks(_preds(), [3, 4, 5])
    s = accuracy.summary(pg, half_life=2.0)
    assert s["predicted"] == pytest.approx(pg["predicted"].sum()) and s["actual"] == pg["actual"].sum()
    assert s["bias"] == pytest.approx((pg["predicted"] - pg["actual"]).mean())
    assert s["mae"] == pytest.approx((pg["predicted"] - pg["actual"]).abs().mean())
    assert s["rank_corr"] == pytest.approx(1.0)            # it always ordered the three right
    assert s["captured"] == pytest.approx(1.0)             # fewer than 11 players: top 11 = everyone
    w = 0.5 ** ((5 - pg["round"]) / 2.0)
    assert s["weighted_mae"] == pytest.approx((w * pg["error"].abs()).sum() / w.sum())


def test_by_gameweek_weights_and_totals():
    pg = accuracy.player_gameweeks(_preds(), [3, 4, 5])
    gw = accuracy.by_gameweek(pg, half_life=2.0)
    assert gw["GW"].tolist() == [3, 4, 5]
    assert gw["weight"].tolist() == pytest.approx([0.5, 0.5 ** 0.5, 1.0])
    assert gw["predicted"].sum() == pytest.approx(pg["predicted"].sum())
    assert gw["actual"].sum() == pg["actual"].sum()


def test_calibration_position_source_and_misses():
    pg = accuracy.player_gameweeks(_preds(), [3, 4, 5])
    cal = accuracy.calibration(pg)
    assert cal["n"].sum() == len(pg)
    assert cal.set_index("bin").loc["6-8", "actual"] == pytest.approx(8.0)
    pos = accuracy.by_position(pg, pd.Series({1: "FWD", 2: "MID", 3: "DEF"}))
    assert set(pos["position"]) == {"DEF", "MID", "FWD"} and pos["player_gameweeks"].sum() == len(pg)

    live = pd.DataFrame({"id": [1, 1, 1], "gw": [3, 4, 5], "pts_goals_scored": [4, 8, 4],
                         "pts_minutes": [2, 4, 2], "pts_own_goals": [0, -2, 0]})
    src = accuracy.by_source(pg, live).set_index("source")
    assert src.at["Goals", "actual"] == 16 and src.at["Appearance", "actual"] == 8
    assert src.at["Goals", "predicted"] == pytest.approx(pg["Goals"].sum())
    assert src.at["Other (own goals, penalties)", "actual"] == -2

    over, under = accuracy.misses(pg, n=2)
    assert over.iloc[0]["id"] == 3 and over.iloc[0]["reason"] == "Didn't play"
    assert under.iloc[0]["id"] == 1
    ms = accuracy.minutes_split(pg)
    assert ms["didnt_play"] == 3 and 0 < ms["share_from_absent"] < 1


def _evaluation():
    """A harness result (model/evaluation.py) built from _preds(): three players, GW3-5."""
    from fpl_optimiser.model import evaluation
    fw = _preds().assign(fixture_id=lambda d: d.index, event_weight=0.5, in_pool=True,
                         opponent_team=9)
    return evaluation.Evaluation(season="2026-27", rounds=[3, 4, 5], first_week=fw, horizons={},
                                 pools={3: {1, 2, 3}, 4: {1, 2, 3}, 5: {1, 2, 3}},
                                 weights={3: 0.5, 4: 0.5, 5: 0.5}, weight_rounds={3: [], 4: [3], 5: [3, 4]})


def _season(ev, live_points=None):
    """A Season-like object whose FPL figures agree with the evaluation's actuals."""
    pg = ev.first_week.groupby(["id", "round"])[["target_points", "minutes"]].sum().reset_index()
    live = pg.rename(columns={"round": "gw"})
    if live_points is not None:
        live = live.assign(target_points=live_points)
    live = live.rename(columns={"target_points": "total_points"}).assign(explained=True)
    history = ev.first_week.rename(columns={"target_points": "total_points"})[["id", "round", "total_points", "minutes"]]
    totals = history.groupby("id")["total_points"].sum()
    rounds = sorted(pg["round"].unique())
    return NS(gameweeks=pd.DataFrame({"id": rounds, "finished": True, "data_checked": True}),
              live=live, history=history, pending=history.iloc[0:0],
              players=pd.DataFrame({"id": totals.index, "total_points": totals.to_numpy()}))


def test_data_checks_pass_when_everything_agrees_and_catch_problems():
    ev = _evaluation()
    ok = accuracy.reconcile(ev, _season(ev))
    assert all(c["ok"] is True for c in ok), [c for c in ok if c["ok"] is not True]

    def failed(check_words, ev=ev, season=None):
        out = accuracy.reconcile(ev, season or _season(ev))
        return next(c for c in out if check_words in c["check"])["ok"] is False

    bad_live = _season(ev, live_points=_season(ev).live["total_points"] + 1)
    assert failed("official gameweek", season=bad_live)
    leaky = _evaluation()
    leaky.first_week["trained_to"] = leaky.first_week["round"]
    assert failed("later data", ev=leaky)
    other_h = _evaluation()
    other_h.first_week["half_life"] = 4.0
    assert failed("half-life", ev=other_h)
    broken = _evaluation()
    broken.first_week["xp_blend"] += 0.1
    assert failed("blend", ev=broken)
    peeking = _evaluation()
    peeking.weight_rounds = {3: [], 4: [3, 4], 5: [3, 4]}
    assert failed("Blend weights", ev=peeking)
    unmatched = _evaluation()
    unmatched.unmatched = 2
    assert failed("match record", ev=unmatched)
    no_pools = _evaluation()
    no_pools.pools = {3: {1}}
    assert failed("Pools", ev=no_pools)

    no_live = _season(ev)
    no_live.live = pd.DataFrame()
    info = next(c for c in accuracy.reconcile(ev, no_live) if "official" in c["check"])
    assert info["ok"] is None


def test_forecasts_are_logged_and_read_back(tmp_path):
    xp = pd.DataFrame({"id": [1, 2, 1], "web_name": ["A", "B", "A"], "gw": [9, 9, 10], "xp": [5.0, 2.0, 4.0],
                       "fixtures": ["ARS (H)", "CHE (A)", "LIV (H)"]})
    fc = NS(season="2026-27", next_gw=9, xp=xp)
    path = forecast_log.save(fc, root=tmp_path)
    assert path.name == "gw09.csv"
    back = forecast_log.load("2026-27", [8, 9], root=tmp_path)
    assert back["id"].tolist() == [1, 2] and back["xp"].tolist() == [5.0, 2.0] and back["logged_at"].notna().all()
    assert forecast_log.load("2026-27", [8], root=tmp_path).empty


def test_half_life_check_and_forecast_path_on_a_filed_store(two_seasons):
    from fpl_optimiser.features import multi_season
    from fpl_optimiser.model import evaluation

    table = multi_season.build_all()
    assert (table["half_life"] == config.RECENCY_HALF_LIFE_GW).all()      # recorded on the table
    ev = evaluation.run(table=table, horizon=1)
    h = config.RECENCY_HALF_LIFE_GW
    hl = accuracy.half_life_check(ev, (h / 2, h))
    assert hl["half_life"].tolist() == [h / 2, h] and hl["current"].tolist() == [False, True]
    pool_rows = ev.first_week[ev.first_week["in_pool"]][["id", "round"]].drop_duplicates()
    assert (hl["player_gameweeks"] == len(pool_rows)).all()               # the same player-gameweeks

    check = accuracy.forecast_path_check(table, ev)
    assert check["ok"] is True, check["detail"]             # training-table path = forecast path, exactly
    pg = accuracy.reference_rows(ev)
    assert (pg["predicted"] == pg["error"] + pg["actual"]).all() and "points_model" in pg
