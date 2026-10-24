"""Check config.py's FPL rules against FPL's own published game settings.

FPL's bootstrap-static carries the live rules: `game_settings` (squad size, budget,
club limit, transfer bank), `game_config.scoring` (points per event, by position)
and `chips`. `check_rules()` lists every place config.py disagrees, so a rule
change (like 2026/27's goalkeeper goal going up to 10 points) is caught on the next
pull instead of silently mis-valuing players.
"""
from __future__ import annotations

import config

_POSITIONS = {"GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
_CHIP_NAMES = {"wildcard": "wildcard", "freehit": "free_hit",
               "bboost": "bench_boost", "3xc": "triple_captain"}

# config.SCORING key -> FPL game_config.scoring key
_BY_POSITION = {"goal": "goals_scored", "clean_sheet": "clean_sheets",
                "goals_conceded_per_2": "goals_conceded"}
_FLAT = {"assist": "assists", "minutes_60_plus": "long_play", "minutes_under_60": "short_play",
         "saves_per_3": "saves", "penalty_save": "penalties_saved",
         "penalty_miss": "penalties_missed", "yellow_card": "yellow_cards",
         "red_card": "red_cards", "own_goal": "own_goals"}


def check_rules(bootstrap: dict) -> list[str]:
    """Every disagreement between config.py and FPL's published rules (empty = all agree)."""
    problems: list[str] = []

    def same(label: str, ours, theirs) -> None:
        if theirs is not None and ours != theirs:
            problems.append(f"{label}: config.py has {ours}, FPL has {theirs}")

    gs = bootstrap.get("game_settings", {})
    same("squad size", config.SQUAD_SIZE, gs.get("squad_squadsize"))
    same("starting XI size", config.XI_SIZE, gs.get("squad_squadplay"))
    same("max players per club", config.MAX_PER_CLUB, gs.get("squad_team_limit"))
    if gs.get("squad_total_spend") is not None:
        same("budget (£m)", config.BUDGET, gs["squad_total_spend"] / 10)
    if gs.get("max_extra_free_transfers") is not None:
        same("max banked free transfers", config.MAX_BANKED_TRANSFERS,
             config.FREE_TRANSFERS_PER_GW + gs["max_extra_free_transfers"])

    scoring = bootstrap.get("game_config", {}).get("scoring", {})
    for ours, theirs in _BY_POSITION.items():
        for fpl_pos, pos in _POSITIONS.items():
            if isinstance(scoring.get(theirs), dict) and fpl_pos in scoring[theirs]:
                same(f"{ours} [{pos}]", config.SCORING[ours][pos], scoring[theirs][fpl_pos])
    for ours, theirs in _FLAT.items():
        same(ours, config.SCORING[ours], scoring.get(theirs))
    defcon = scoring.get("defensive_contribution")
    if isinstance(defcon, dict):
        for fpl_pos, pos in _POSITIONS.items():
            if pos in config.SCORING["defensive_contribution_threshold"] and fpl_pos in defcon:
                same(f"defensive contribution points [{pos}]",
                     config.SCORING["defensive_contribution_points"], defcon[fpl_pos])

    chips = bootstrap.get("chips", [])
    if chips:
        same("chips", sorted(config.CHIPS), sorted({_CHIP_NAMES.get(c["name"], c["name"]) for c in chips}))
        same("chip sets per season", config.CHIP_SETS_PER_SEASON,
             sum(c["name"] == "wildcard" for c in chips))
        same("first chip set expires after GW", config.CHIP_SPLIT_GW,
             min(c["stop_event"] for c in chips))
    return problems
