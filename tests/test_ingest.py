"""Ingestion parsing tests — run on mock payloads, no network required."""
import json

import pandas as pd

from fpl_optimiser.ingest import fpl_api, fbref, understat, team_map

# --------------------------------------------------------------------------- #
# FPL API
# --------------------------------------------------------------------------- #
MOCK_BOOTSTRAP = {
    "teams": [
        {"id": 1, "name": "Arsenal", "short_name": "ARS", "strength": 5,
         "strength_attack_home": 1300, "strength_attack_away": 1320,
         "strength_defence_home": 1290, "strength_defence_away": 1310},
        {"id": 2, "name": "Man City", "short_name": "MCI", "strength": 5,
         "strength_attack_home": 1350, "strength_attack_away": 1340,
         "strength_defence_home": 1300, "strength_defence_away": 1305},
    ],
    "elements": [
        {"id": 100, "web_name": "Haaland", "first_name": "Erling",
         "second_name": "Haaland", "team": 2, "element_type": 4,
         "now_cost": 151, "total_points": 200, "minutes": 2800,
         "goals_scored": 27, "assists": 5, "expected_goals": "24.6",
         "expected_assists": "4.1", "selected_by_percent": "55.0",
         "ict_index": "210.5"},
        {"id": 101, "web_name": "Saliba", "first_name": "William",
         "second_name": "Saliba", "team": 1, "element_type": 2,
         "now_cost": 60, "total_points": 150, "minutes": 3100,
         "clearances_blocks_interceptions": 180, "tackles": 40,
         "recoveries": 210, "expected_goals_conceded": "30.2",
         "selected_by_percent": "30.0", "ict_index": "120.0"},
    ],
    "events": [
        {"id": 1, "name": "Gameweek 1", "deadline_time": "2026-08-22T10:00:00Z",
         "finished": True, "is_previous": False, "is_current": False, "is_next": True,
         "average_entry_score": 55, "highest_score": 120},
    ],
}

MOCK_FIXTURES = [
    {"id": 1, "event": 1, "team_h": 1, "team_a": 2,
     "kickoff_time": "2026-08-22T14:00:00Z", "finished": True,
     "team_h_score": 2, "team_a_score": 1,
     "team_h_difficulty": 4, "team_a_difficulty": 3},
]


def test_parse_players_price_position_and_team():
    df = fpl_api.parse_players(MOCK_BOOTSTRAP).set_index("id")
    assert df.loc[100, "price"] == 15.1
    assert df.loc[100, "position"] == "FWD"
    assert df.loc[101, "position"] == "DEF"
    assert df.loc[100, "team_name"] == "Man City"
    assert abs(df.loc[100, "expected_goals"] - 24.6) < 1e-9


def test_parse_players_fills_missing_defensive_cols():
    df = fpl_api.parse_players(MOCK_BOOTSTRAP).set_index("id")
    assert pd.isna(df.loc[100, "tackles"])
    assert df.loc[101, "tackles"] == 40


def test_parse_fixtures_team_names_and_fdr():
    fx = fpl_api.parse_fixtures(MOCK_FIXTURES, fpl_api.parse_teams(MOCK_BOOTSTRAP))
    row = fx.iloc[0]
    assert row["home"] == "Arsenal" and row["away"] == "Man City"
    assert row["home_fdr"] == 4


def test_parse_gameweeks_next_flag():
    gw = fpl_api.parse_gameweeks(MOCK_BOOTSTRAP)
    assert bool(gw.loc[gw["id"] == 1, "is_next"].iloc[0]) is True


# --------------------------------------------------------------------------- #
# Understat (primary advanced source)
# --------------------------------------------------------------------------- #
def _mock_understat_page() -> str:
    """Build a page exactly as Understat embeds it: \\xHH-escaped UTF-8 JSON."""
    sample = {
        "11": {"id": "11", "title": "Arsenal", "history": [
            {"h_a": "h", "xG": 1.5, "xGA": 0.4, "npxG": 1.3, "npxGA": 0.4,
             "ppda": {"att": 200, "def": 20}, "ppda_allowed": {"att": 150, "def": 10},
             "deep": 8, "deep_allowed": 2, "scored": 2, "missed": 0,
             "xpts": 2.3, "result": "w", "date": "2025-08-16 15:00:00"},
        ]},
        "13": {"id": "13", "title": "Burnley", "history": [
            {"h_a": "a", "xG": 0.5, "xGA": 2.1, "npxG": 0.5, "npxGA": 2.1,
             "ppda": {"att": 300, "def": 15}, "ppda_allowed": {"att": 120, "def": 30},
             "deep": 1, "deep_allowed": 9, "scored": 0, "missed": 2,
             "xpts": 0.3, "result": "l", "date": "2025-08-16 15:00:00"},
        ]},
    }
    js = json.dumps(sample)
    escaped = "".join("\\x%02x" % b for b in js.encode("utf-8"))
    return f"<script>var teamsData = JSON.parse('{escaped}');</script>"


def test_understat_parse_extracts_and_decodes():
    df = understat.parse_team_matches(_mock_understat_page()).set_index("team")
    assert df.loc["Arsenal", "xga"] == 0.4
    assert df.loc["Burnley", "xga"] == 2.1
    # ppda = att / def  => Arsenal 200/20 = 10.0
    assert abs(df.loc["Arsenal", "ppda"] - 10.0) < 1e-9
    # 'missed' is goals conceded
    assert df.loc["Burnley", "missed"] == 2


def test_understat_season_code():
    assert understat.to_understat_season("2025-26") == "2025"
    assert understat.to_understat_season("2024-25") == "2024"


# --------------------------------------------------------------------------- #
# helpers that don't need network
# --------------------------------------------------------------------------- #
def test_soccerdata_season_code():
    assert fbref.to_soccerdata_season("2025-26") == "2526"


def test_team_reconcile_matches_variants():
    fpl = ["Man City", "Tottenham", "Nott'm Forest", "Newcastle"]
    fbref_names = ["Manchester City", "Tottenham Hotspur",
                   "Nottingham Forest", "Newcastle Utd"]
    rec = team_map.reconcile(fpl, fbref_names)
    assert rec["matched"]["Man City"] == "Manchester City"
    assert rec["fpl_only"] == [] and rec["fbref_only"] == []


def test_team_reconcile_understat_names():
    fpl = ["Man Utd", "Wolves", "Spurs"]
    us = ["Manchester United", "Wolverhampton Wanderers", "Tottenham"]
    rec = team_map.reconcile(fpl, us)
    assert rec["matched"]["Man Utd"] == "Manchester United"
    assert rec["matched"]["Wolves"] == "Wolverhampton Wanderers"
    assert rec["matched"]["Spurs"] == "Tottenham"
    assert rec["fpl_only"] == [] and rec["fbref_only"] == []