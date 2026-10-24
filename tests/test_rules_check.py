"""config.py vs FPL's published rules — on a mock bootstrap (no network)."""
import copy

from fpl_optimiser.ingest import rules_check

AGREES = {
    "game_settings": {"squad_squadsize": 15, "squad_squadplay": 11, "squad_team_limit": 3,
                      "squad_total_spend": 1000, "max_extra_free_transfers": 4},
    "game_config": {"scoring": {
        "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}, "assists": 3,
        "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
        "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
        "long_play": 2, "short_play": 1, "saves": 1, "penalties_saved": 5,
        "penalties_missed": -2, "yellow_cards": -1, "red_cards": -3, "own_goals": -2,
        "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2}}},
    "chips": [{"name": n, "start_event": s, "stop_event": e}
              for n in ("wildcard", "freehit", "bboost", "3xc") for s, e in ((1, 19), (20, 38))],
}


def test_config_matches_published_2026_27_rules():
    assert rules_check.check_rules(AGREES) == []


def test_rule_changes_are_reported():
    changed = copy.deepcopy(AGREES)
    changed["game_config"]["scoring"]["goals_scored"]["GKP"] = 6
    changed["game_settings"]["max_extra_free_transfers"] = 2
    problems = rules_check.check_rules(changed)
    assert any("goal [GK]" in p and "FPL has 6" in p for p in problems)
    assert any("banked free transfers" in p and "FPL has 3" in p for p in problems)


def test_missing_sections_are_not_errors():
    assert rules_check.check_rules({}) == []
