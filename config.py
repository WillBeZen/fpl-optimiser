"""
config.py — single source of truth for FPL rules and strategy knobs.

Two clearly separated blocks:

  RULES     The game's hard constraints — FACTS about how FPL scores and which
            squads are legal. The optimiser treats these as inviolable. The
            baseline below is the 2025/26 rule set (the most recent *confirmed*
            season). The 2026/27 rules are not published yet (FPL usually
            announces in late July); when they drop, edit THIS block only.

  STRATEGY  Your tunable dials. Nothing here is a rule — these are choices that
            trade risk against reward. Change them freely and re-run; the
            backtest (Section 3) tells you whether a change actually helps.

This file imports no third-party packages, so it always loads.
Validate it with:  python config.py   (or)   pytest tests/test_config.py
"""

from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Seasons
# --------------------------------------------------------------------------- #
# We TRAIN and BACKTEST on the just-completed season, then go LIVE on the next.
SEASON_TRAIN = "2025-26"   # completed — used to fit and validate the models
SEASON_LIVE = "2026-27"    # starts 22 Aug 2026 — what we actually pick teams for

GAMEWEEKS_IN_SEASON = 38

# --------------------------------------------------------------------------- #
# RULES — FPL hard constraints (2025/26 baseline; confirm for 2026/27)
# --------------------------------------------------------------------------- #

# Squad composition: exactly 15 players in these position quotas (sums to 15).
SQUAD_SIZE = 15
POSITION_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Money. The FPL API stores prices in tenths of a million (e.g. 75 = £7.5m);
# we keep BUDGET in £m here and convert at the ingestion boundary.
BUDGET = 100.0          # £m to assemble the initial 15
MAX_PER_CLUB = 3        # at most 3 players from any single Premier League club

# Starting XI: pick 11 of the 15. The formation must satisfy these (min, max)
# bounds, with exactly 1 goalkeeper on the pitch.
XI_SIZE = 11
FORMATION_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}

# Transfers.
FREE_TRANSFERS_PER_GW = 1   # you earn 1 free transfer each gameweek...
MAX_BANKED_TRANSFERS = 5    # ...but can stockpile at most 5 unused at once
TRANSFER_HIT_COST = 4       # each transfer beyond your free allowance costs 4 pts

# Captaincy.
CAPTAIN_MULTIPLIER = 2          # captain scores double
VICE_CAPTAIN_MULTIPLIER = 1     # vice only matters if the captain doesn't play

# Chips (2025/26 structure). TWO full sets across the season; the FIRST set must
# be used on/before the split-gameweek deadline, the SECOND set after it.
CHIPS = ("wildcard", "free_hit", "bench_boost", "triple_captain")
CHIP_SETS_PER_SEASON = 2
CHIP_SPLIT_GW = 19              # first-half chips expire at the GW19 deadline
TRIPLE_CAPTAIN_MULTIPLIER = 3

# Scoring table (points). The prediction model uses this to turn predicted
# events (goals, assists, clean sheets, ...) into predicted FPL points.
SCORING = {
    "minutes_under_60": 1,
    "minutes_60_plus": 2,
    "goal": {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": 3,                                       # all positions
    "clean_sheet": {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0},   # needs 60+ mins
    "goals_conceded_per_2": {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0},
    "saves_per_3": 1,          # goalkeepers: 1 pt per 3 saves
    "penalty_save": 5,
    "penalty_miss": -2,
    "yellow_card": -1,
    "red_card": -3,
    "own_goal": -2,
    "bonus_max": 3,            # 1–3 pts to the top BPS performers in a match
    # New in 2025/26 — "defensive contributions" (CBIT for DEF; CBIRT incl. ball
    # recoveries for MID/FWD). Reaching the threshold in a match earns the points.
    "defensive_contribution_points": 2,
    "defensive_contribution_threshold": {"DEF": 10, "MID": 12, "FWD": 12},
}

# --------------------------------------------------------------------------- #
# STRATEGY — tunable dials (not rules; change and re-backtest)
# --------------------------------------------------------------------------- #

# Recency weighting (Section 2). A match played `g` gameweeks ago is weighted by
# 0.5 ** (g / HALF_LIFE): at g = HALF_LIFE it counts half as much as a current
# match; at 2*HALF_LIFE, a quarter. Smaller half-life = more reactive to form.
RECENCY_HALF_LIFE_GW = 8.0

# Multi-week planning (Section 5). How many gameweeks ahead the planner reasons
# about when choosing to bank a transfer, spend it, or take a hit.
PLANNING_HORIZON_GW = 5

# Hit threshold (Section 5). Take a -4 hit only if the EXTRA points it buys over
# the planning horizon exceed this. Deliberately > TRANSFER_HIT_COST so a
# transfer must clearly pay for itself, not merely break even on paper (the gap
# is our margin for prediction error).
HIT_THRESHOLD_POINTS = 6.0

# Opponent archetypes (Section 2). Number of defensive "styles" the 20 teams are
# clustered into, for interpretation and for Claude's narrative.
ARCHETYPE_K = 3

# Minutes floor (Sections 3–4). Players below this predicted start probability
# are treated as bench fodder (not picked to start).
MIN_START_PROB = 0.6

# Advanced opponent stats source. "understat" is browserless (xGA, PPDA, deep
# completions) and is the primary source. "fbref" is optional and needs Google
# Chrome installed — soccerdata drives a headless browser because FBref is behind
# Cloudflare.
ADVANCED_STATS_SOURCE = "understat"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"          # soccerdata + FPL response cache

FPL_API_BASE = "https://fantasy.premierleague.com/api"

# --------------------------------------------------------------------------- #
# Validation — fail loudly if the rules are internally inconsistent.
# --------------------------------------------------------------------------- #
def validate() -> None:
    assert sum(POSITION_QUOTA.values()) == SQUAD_SIZE, \
        "position quota must sum to squad size"
    lo_total = sum(lo for lo, _ in FORMATION_BOUNDS.values())
    hi_total = sum(hi for _, hi in FORMATION_BOUNDS.values())
    assert lo_total <= XI_SIZE <= hi_total, \
        "no legal formation can field exactly the XI size"
    for pos, (lo, _) in FORMATION_BOUNDS.items():
        assert lo <= POSITION_QUOTA[pos], f"{pos} formation min exceeds squad quota"
    assert 0 < MAX_PER_CLUB <= SQUAD_SIZE
    assert HIT_THRESHOLD_POINTS >= TRANSFER_HIT_COST, \
        "hit threshold should be >= the hit cost"
    assert 0 <= MIN_START_PROB <= 1
    assert RECENCY_HALF_LIFE_GW > 0
    assert 1 <= PLANNING_HORIZON_GW <= GAMEWEEKS_IN_SEASON


if __name__ == "__main__":
    validate()
    print("config OK")
