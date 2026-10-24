"""Feature-engineering tests (Section 2) — small synthetic season, no network."""
import math

import numpy as np
import pandas as pd
import pytest

from fpl_optimiser.features import build, opponent, recency

HL = 8.0  # recency half-life used throughout

# The column contract Section 3 trains on (see docs/02_feature_engineering.md).
EXPECTED_COLUMNS = [
    "id", "round", "web_name", "position", "was_home", "opponent_team",
    "target_points", "minutes",
    "form_points_pg", "form_minutes_pg", "form_xg_pg", "form_xa_pg", "form_xgi_pg",
    "form_defcon_pg", "form_bonus_pg", "form_bps_pg", "games_played", "form_starts_rate",
    "opp_gc_pg_asof", "opp_archetype", "opp_str_def", "x_xgi_oppgc", "is_cold",
]

# --------------------------------------------------------------------------- #
# A tiny season: three teams; team 3 plays twice in GW3 (a double gameweek);
# GW5 hasn't been played yet.
# --------------------------------------------------------------------------- #
FIXTURES = pd.DataFrame({
    "gameweek":   [1, 2, 3, 3, 4, 5],
    "finished":   [True, True, True, True, True, False],
    "home_id":    [1, 2, 1, 2, 3, 1],
    "away_id":    [2, 1, 3, 3, 1, 2],
    "home_score": [2, 1, 0, 1, 2, None],
    "away_score": [0, 3, 0, 1, 2, None],
})

PLAYERS = pd.DataFrame({
    "id": [10, 20, 30, 40],
    "web_name": ["Mid", "Def", "Fwd", "Def2"],
    "position": ["MID", "DEF", "FWD", "DEF"],
    "team": [1, 3, 2, 1],
})

PROFILES = pd.DataFrame({
    "team_id": [1, 2, 3],
    "team": ["A", "B", "C"],
    "gc_pg": [0.75, 2.0, 1.0],
    "str_def_home": [1300, 1050, 1200],
    "str_def_away": [1250, 1000, 1150],
    "str_overall": [5, 2, 4],
    "archetype": [0, 2, 1],
})

_HISTORY_COLS = ["id", "round", "opponent_team", "was_home", "minutes", "total_points",
                 "expected_goals", "expected_assists", "expected_goal_involvements",
                 "defensive_contribution", "bonus", "bps"]


def _history() -> pd.DataFrame:
    """Per-match rows as FPL's element-summary gives them (xG etc. arrive as strings)."""
    rows = [
        (10, 1, 2, True,  90, 6, "0.50", "0.10", "0.60",  3, 1, 20),
        (10, 2, 2, False, 70, 2, "0.10", "0.00", "0.10",  5, 0, 10),
        (10, 3, 3, True,   0, 0, "0.00", "0.00", "0.00",  0, 0,  0),
        (10, 4, 3, False, 90, 9, "0.90", "0.30", "1.20",  4, 3, 35),
        (20, 3, 1, False, 90, 1, "0.00", "0.00", "0.00", 11, 0, 15),   # debut in a DGW:
        (20, 3, 2, False, 90, 6, "0.20", "0.00", "0.20", 12, 2, 28),   # two GW3 matches
        (20, 4, 1, True,  90, 2, "0.10", "0.10", "0.20",  9, 0, 18),
        (30, 1, 1, False, 90, 2, None,   "0.00", None,    1, 0,  8),   # missing xG
        (30, 2, 1, True,  45, 1, "0.30", "0.00", "0.30",  0, 0,  4),
        (30, 3, 3, True,  90, 5, "0.70", "0.20", "0.90",  2, 1, 22),
        (40, 1, 2, True,  90, 8, "0.10", "0.05", "0.15", 12, 2, 30),
        (40, 2, 2, False, 90, 1, "0.00", "0.00", "0.00", 10, 0, 12),
    ]
    return pd.DataFrame(rows, columns=_HISTORY_COLS)


def _table(history=None, fixtures=FIXTURES) -> pd.DataFrame:
    return build.build_training_table(_history() if history is None else history,
                                      fixtures, PLAYERS, PROFILES, HL)


# --------------------------------------------------------------------------- #
# recency
# --------------------------------------------------------------------------- #
def test_decay_halves_at_half_life():
    assert recency.decay_weight(0, HL) == 1.0
    assert recency.decay_weight(HL, HL) == pytest.approx(0.5)
    assert recency.decay_weight(2 * HL, HL) == pytest.approx(0.25)


def test_weighted_mean_favours_recent_and_ignores_nan():
    # 10 pts last week vs 0 pts ten weeks ago: well above the flat mean of 5
    assert recency.weighted_mean([10, 0], [1, 10], HL) > 5
    assert recency.weighted_mean([4, np.nan], [1, 2], HL) == 4
    assert math.isnan(recency.weighted_mean([np.nan], [1], HL))


# --------------------------------------------------------------------------- #
# opponent profiles + archetypes
# --------------------------------------------------------------------------- #
def _six_teams(**overrides) -> pd.DataFrame:
    df = pd.DataFrame({
        "team_id": range(1, 7),
        "team": list("ABCDEF"),
        "gc_pg": [0.6, 0.7, 1.4, 1.5, 2.3, 2.4],
        "str_def_home": [1350, 1340, 1150, 1140, 1000, 990],
        "str_def_away": [1340, 1330, 1140, 1130, 990, 980],
        "str_overall": [5, 5, 3, 3, 2, 2],
    })
    return df.assign(**overrides)


def test_archetypes_order_by_goals_conceded():
    arch = opponent.assign_archetypes(_six_teams(), k=3).set_index("team")["archetype"]
    assert arch["A"] == arch["B"] == 0          # meanest defences
    assert arch["C"] == arch["D"] == 1
    assert arch["E"] == arch["F"] == 2          # leakiest


def test_archetypes_tolerate_a_feature_missing_for_every_team():
    arch = opponent.assign_archetypes(_six_teams(str_overall=np.nan), k=3)
    assert sorted(arch["archetype"].unique()) == [0, 1, 2]


def test_build_opponent_profiles_joins_on_reconciled_names():
    opp = pd.DataFrame({"team": ["Manchester City", "Arsenal"],
                        "goals_conceded_per_game": [1.1, 0.7]})
    teams = pd.DataFrame({"id": [1, 13], "name": ["Arsenal", "Man City"], "strength": [5, 5],
                          "strength_defence_home": [1300, 1320],
                          "strength_defence_away": [1310, 1330]})
    prof = opponent.build_opponent_profiles(opp, teams).set_index("team")
    assert prof.loc["Man City", "gc_pg"] == 1.1
    assert prof.loc["Arsenal", "team_id"] == 1


# --------------------------------------------------------------------------- #
# as-of (leakage-free) building blocks
# --------------------------------------------------------------------------- #
def test_weighted_form_asof_excludes_current_and_future_rounds():
    h = _history().query("id == 10")
    form = build.weighted_form_asof(h, 3, HL)
    w1, w2 = recency.decay_weight([2, 1], HL)              # GW1 and GW2, seen from GW3
    assert form["form_points_pg"] == pytest.approx((6 * w1 + 2 * w2) / (w1 + w2))
    assert form["games_played"] == 2

    # rewriting GW3 and later must not change GW3's form
    h2 = h.copy()
    h2.loc[h2["round"] >= 3, "total_points"] = 99
    assert build.weighted_form_asof(h2, 3, HL) == form


def test_team_defence_asof_uses_prior_rounds_only():
    d = build.team_defence_asof(FIXTURES, HL).set_index(["team_id", "round"])["opp_gc_pg_asof"]
    assert math.isnan(d[(1, 1)])                    # nothing known before GW1
    assert d[(1, 2)] == 0.0                         # GW1's 0 conceded, not GW2's 1
    w1, w2 = recency.decay_weight([2, 1], HL)
    assert d[(1, 3)] == pytest.approx((0 * w1 + 1 * w2) / (w1 + w2))

    # changing a GW3 result can only move later rounds
    fx = FIXTURES.copy()
    fx.loc[fx["gameweek"] == 3, "home_score"] = 7
    d2 = build.team_defence_asof(fx, HL).set_index(["team_id", "round"])["opp_gc_pg_asof"]
    upto_gw3 = d.index.get_level_values("round") <= 3
    pd.testing.assert_series_equal(d[upto_gw3], d2[upto_gw3])


def test_team_defence_asof_double_gameweek_and_unplayed():
    d = build.team_defence_asof(FIXTURES, HL)
    assert not d.duplicated(["team_id", "round"]).any()     # one row per (team, round)
    team3 = d[d["team_id"] == 3].set_index("round")["opp_gc_pg_asof"]
    assert list(team3.index) == [3, 4]
    assert math.isnan(team3[3])
    assert team3[4] == pytest.approx(0.5)     # both GW3 matches (0 and 1 conceded) count equally
    assert 5 not in d["round"].values         # the unplayed GW5 contributes nothing


def test_team_defence_asof_before_any_fixture_finishes():
    d = build.team_defence_asof(FIXTURES.assign(finished=False), HL)
    assert d.empty
    assert list(d.columns) == ["team_id", "round", "opp_gc_pg_asof"]


# --------------------------------------------------------------------------- #
# the assembled training table
# --------------------------------------------------------------------------- #
def test_training_table_smoke():
    tbl = _table()
    assert list(tbl.columns) == EXPECTED_COLUMNS
    assert len(tbl) == len(_history())              # one row per match (a DGW gives two)
    assert tbl["target_points"].sum() == _history()["total_points"].sum()
    assert tbl.loc[tbl["round"] == 1, "is_cold"].all()


def test_training_table_matches_reference_form():
    """The vectorised build gives weighted_form_asof's numbers for every row."""
    hist, tbl = _history(), _table()
    for _, row in tbl.iterrows():
        ref = build.weighted_form_asof(hist[hist["id"] == row["id"]], row["round"], HL)
        for col, want in ref.items():
            if pd.isna(want):
                continue                            # cold-start prior (tested below)
            assert row[col] == pytest.approx(want), (row["id"], row["round"], col)


def test_training_table_opponent_features_are_venue_correct():
    tbl = _table().set_index(["id", "round", "opponent_team"])
    # player 10 at HOME to team 2 -> team 2 played away -> its AWAY rating
    assert tbl.loc[(10, 1, 2), "opp_str_def"] == 1000
    assert tbl.loc[(10, 2, 2), "opp_str_def"] == 1050      # away at team 2 -> its home rating
    assert tbl.loc[(10, 1, 2), "opp_archetype"] == 2

    d = build.team_defence_asof(FIXTURES, HL).set_index(["team_id", "round"])["opp_gc_pg_asof"]
    row = tbl.loc[(10, 4, 3)]
    assert row["opp_gc_pg_asof"] == pytest.approx(d[(3, 4)])
    assert row["x_xgi_oppgc"] == pytest.approx(row["form_xgi_pg"] * d[(3, 4)])


def test_training_table_has_no_leakage():
    """Rewriting what happened in GW3 (incl. its double gameweek) must not change a
    single feature of any GW1-GW3 row; only GW3's targets and later rows may move."""
    base = _table()
    hist = _history()
    gw3 = hist["round"] == 3
    hist.loc[gw3, "total_points"] = 99
    hist.loc[gw3, "bps"] = 99
    hist.loc[gw3, "minutes"] = 1
    fx = FIXTURES.copy()
    fx.loc[fx["gameweek"] == 3, ["home_score", "away_score"]] = 9
    changed = _table(hist, fx)

    features = [c for c in EXPECTED_COLUMNS if c not in ("target_points", "minutes")]
    upto_gw3 = base["round"] <= 3
    pd.testing.assert_frame_equal(base.loc[upto_gw3, features], changed.loc[upto_gw3, features])
    assert not base.loc[~upto_gw3, features].equals(changed.loc[~upto_gw3, features])


def test_cold_start_prior_uses_earlier_rounds_only():
    tbl = _table().set_index(["id", "round", "opponent_team"])
    # nothing is known before GW1: a GW1 row stays NaN and is flagged cold
    assert tbl.loc[(10, 1, 2), "is_cold"]
    assert math.isnan(tbl.loc[(10, 1, 2), "form_points_pg"])
    # a later debut borrows its position's median form from EARLIER rounds: the only
    # DEF form known before GW3 is player 40's GW2 row (= his GW1 haul of 8)
    for opp in (1, 2):                               # both of player 20's GW3 matches
        assert tbl.loc[(20, 3, opp), "is_cold"]
        assert tbl.loc[(20, 3, opp), "form_points_pg"] == 8


def test_training_table_empty_history():
    tbl = build.build_training_table(pd.DataFrame(), FIXTURES, PLAYERS, PROFILES, HL)
    assert tbl.empty and list(tbl.columns) == EXPECTED_COLUMNS
