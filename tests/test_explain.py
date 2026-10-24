"""Model explainers — the numbers behind the report's "The model" tab (no network)."""
import numpy as np
import pandas as pd
import pytest

from fpl_optimiser.features import build, multi_season
from fpl_optimiser.model import explain, predict
from tests.test_multi_season import two_seasons  # noqa: F401  (pytest fixture)


def test_weight_curve_halves_every_half_life_and_shares_add_up():
    c = explain.weight_curve(as_of_round=9, half_life=4.0).set_index("label")
    assert c.loc["GW8", "weight"] == pytest.approx(0.5 ** (1 / 4))
    assert c.loc["GW5", "weight"] == pytest.approx(0.5)               # 4 gameweeks ago
    assert c.loc["GW1", "weight"] == pytest.approx(0.25)              # 8 ago
    assert c["share"].sum() == pytest.approx(1.0) and c.loc["Last season", "share"] > 0


def test_last_season_fades_through_the_season():
    s = explain.prior_share()
    assert s["share"].iloc[0] == pytest.approx(1.0)                   # GW1: all last season
    assert s["share"].is_monotonic_decreasing and s["share"].iloc[-1] < 0.05


def _history():
    return pd.DataFrame({"id": 7, "round": [1, 2, 3, 4], "opponent_team": [2, 3, 2, 3],
                         "was_home": [True, False, True, False], "minutes": [90, 90, 90, 90],
                         "total_points": [2, 2, 2, 14], "expected_goal_involvements": [0.1] * 4})


def test_form_story_weights_recent_matches_more():
    m, s = explain.form_story(_history(), 7, as_of_round=5, half_life=2.0)
    assert m["weight"].is_monotonic_increasing and m["share"].sum() == pytest.approx(1.0)
    assert s["plain_average"] == pytest.approx(5.0)
    assert s["form"] > s["plain_average"]                              # the 14 is the latest match
    assert s["form"] == pytest.approx(build.weighted_form_asof(_history(), 5, 2.0)["form_points_pg"])
    assert s["last_season_share"] == 0 and np.isnan(s["last_season"])

    prior = pd.DataFrame({"id": [7], "pg_total_points": [6.0], "pg_minutes": [90.0], "pg_played_60": [1.0]})
    m2, s2 = explain.form_story(_history(), 7, 5, prior=prior, half_life=2.0)
    assert 0 < s2["last_season_share"] < 1 and s2["last_season"] == 6.0
    assert m2["share"].sum() + s2["last_season_share"] == pytest.approx(1.0)


def test_half_life_table_reacts_faster_when_shorter():
    t = explain.half_life_table(_history(), [7], 5, half_lives=(2.0, 8.0, 32.0)).loc[7]
    assert t["half-life 2"] > t["half-life 8"] > t["half-life 32"] > 5.0


def test_explainers_on_a_filed_store(two_seasons):
    fc = predict.forecast(horizon=3)
    cur, prior, profiles = multi_season.season_inputs(fc.season)
    table = multi_season.build_all()

    fi = explain.feature_importance(fc.model, table, n_repeats=1)
    assert set(fi["feature"]) == set(explain.inputs.FEATURES) and fi["label"].notna().all()

    rows = fc.per_fixture[fc.per_fixture["round"] == fc.next_gw]
    resp = explain.opponent_response(fc.model, rows, grid=[0.8, 1.6])
    assert set(resp["opp_gc_pg"]) == {0.8, 1.6} and resp["pts_if_play"].notna().all()

    ids = list(cur.players["id"])
    grid = explain.opponent_matrix(fc, ids, profiles, cur.fixtures, cur.teams)
    assert set(grid.index) == set(ids) and set(grid.columns) <= set(cur.teams["short_name"])
    star = cur.players.set_index("web_name").loc["Star", "id"]
    assert np.isnan(grid.loc[star, "A"])                               # never plays his own club
    assert (grid.loc[star].dropna() > 0).all()

    types = pd.Series("Type X", index=cur.players["id"])
    mh = explain.matchup_history(table[table["season"] == fc.season], types, profiles)
    assert mh["matches"].sum() == (table[(table["season"] == fc.season)]["minutes"] > 0).sum()
    assert set(mh["style"]) <= set(explain.style_names(profiles).values())

    flags = explain.flag_impact(fc)
    assert (flags["cut"] > 0).all()
