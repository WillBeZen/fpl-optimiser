"""Player profiles + archetypes (the report's player-type graphics) — synthetic data."""
import numpy as np
import pandas as pd
import pytest

from fpl_optimiser.features import player_profiles as pp


def _players(rows):
    cols = ["id", "code", "web_name", "team_name", "position", "price", "total_points",
            "minutes", "expected_goals", "expected_assists", "expected_goal_involvements",
            "defensive_contribution", "saves", "expected_goals_conceded", "bonus"]
    return pd.DataFrame(rows, columns=cols)


def test_profiles_blend_last_season_per_minute():
    curr = _players([(10, 1001, "Star", "ARS", "MID", 8.0, 50, 900, 3.0, 1.0, 4.0, 30, 0, 9.0, 5),
                     (11, 2001, "New", "HUL", "FWD", 6.0, 20, 450, 2.0, 0.5, 2.5, 5, 0, 4.0, 1)])
    prev = _players([(1, 1001, "Star", "ARS", "MID", 7.5, 150, 1800, 6.0, 2.0, 8.0, 60, 0, 18.0, 10)])
    links = pd.DataFrame({"id_prev": [1], "id_curr": [10]})
    p = pp.build_profiles(curr, curr_gameweeks=10, prev=prev, links=links,
                          prev_gameweeks=38, prev_weight=0.5).set_index("id")

    # Star: (3 + 0.5*6) xG over (900 + 0.5*1800) minutes = 6 / 1800 -> 0.3 per 90
    assert p.loc[10, "xg_90"] == pytest.approx(0.3)
    assert p.loc[10, "profile_minutes"] == 1800 and p.loc[10, "has_prev_season"]
    assert p.loc[10, "minutes_share"] == pytest.approx(1800 / (10 * 90 + 0.5 * 38 * 90))
    # New: this season only
    assert p.loc[11, "xg_90"] == pytest.approx(2.0 / 450 * 90)
    assert not p.loc[11, "has_prev_season"]
    assert p.loc[11, "points_per_m"] == pytest.approx(20 / 6.0)


def _midfielders():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):                                   # 10 attackers, 10 ball-winners
        attacker = i < 10
        xg = (6 if attacker else 0.5) + rng.normal(0, 0.3)
        defcon = (20 if attacker else 110) + rng.normal(0, 4)
        rows.append((i, 5000 + i, f"M{i}", "X", "MID", 6.0, 120 if attacker else 60, 2000,
                     xg, 2.0, xg + 2.0, defcon, 0, 20.0, 10 if attacker else 3))
    rows.append((99, 5999, "Cameo", "X", "MID", 4.5, 2, 100, 0.0, 0.0, 0.0, 1, 0, 1.0, 0))
    return _players(rows)


def test_archetypes_split_by_style_and_rank_by_points():
    prof = pp.build_profiles(_midfielders(), curr_gameweeks=30)
    players, types = pp.assign_archetypes(prof, k={"GK": 2, "DEF": 2, "MID": 2, "FWD": 2},
                                          min_minutes=450)
    assert 99 not in set(players["id"])                   # 100 minutes: too few to profile
    a = players.set_index("id")["archetype"]
    assert len(set(a[:10])) == 1 and len(set(a[10:])) == 1 and a[0] != a[10]
    assert a[0] == 0                                      # the higher-scoring type is archetype 0

    mids = types[types["position"] == "MID"].set_index("archetype")
    assert "High xG" in mids.loc[0, "archetype_name"]
    assert "High def. contributions" in mids.loc[1, "archetype_name"]
    assert mids["n_players"].sum() == 20
