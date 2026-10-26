"""
config.py — single source of truth for FPL rules and strategy knobs.

Two clearly separated blocks:

  RULES     The game's hard constraints — FACTS about how FPL scores and which
            squads are legal. The optimiser treats these as inviolable. These are
            the 2026/27 rules, confirmed against FPL's own published game_config;
            ingest/rules_check.py re-checks them after every pull and warns if FPL
            changes anything. If it does, edit THIS block only.

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
# RULES — FPL hard constraints (2026/27, confirmed from FPL's game_config)
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

# Chips (2026/27, as 2025/26). TWO full sets across the season; the FIRST set must
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
    "goal": {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4},    # GK goal = 10 (FPL game_config 2026/27)
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
# Chosen on 2025/26 (scripts/study_last_season.py: every deadline replayed at 4, 6, 8,
# 12 and 16): 4 had the lowest RMSE, better than 8 in 30 of 37 gameweeks, with no
# difference in picking. Checked on 2026/27 GW2-7: lower RMSE than 8 (docs/08).
RECENCY_HALF_LIFE_GW = 4.0

# Multi-week planning (Section 5). How many gameweeks ahead the planner reasons
# about when choosing to bank a transfer, spend it, or take a hit.
PLANNING_HORIZON_GW = 6

# Hit threshold (Section 5). Take a -4 hit only if the EXTRA points it buys over
# the planning horizon exceed this. Deliberately > TRANSFER_HIT_COST so a
# transfer must clearly pay for itself, not merely break even on paper (the gap
# is our margin for prediction error).
HIT_THRESHOLD_POINTS = 6.0

# Horizon decay (Section 5). Each gameweek further ahead counts this much less in
# the plan (0.9: next GW 1.0, then 0.9, 0.81, ...). Predictions get less certain
# with distance, so near-term points are preferred over equal distant ones.
HORIZON_DECAY = 0.9

# Value of a banked free transfer at the end of the plan (Section 5), in expected
# points. Without it the planner would spend every transfer in the last week for
# any tiny gain, because the plan can't see the flexibility it throws away.
FREE_TRANSFER_VALUE = 1.5

# Chip thresholds (Section 5). The planner plays a chip only if it gains more than
# this over the same plan without it, in that gameweek's expected points (the plan
# decays both alike, so a double gameweek a month away can still earn the chip). It
# is the chip's opportunity value: roughly what it should be worth if saved for a
# better week (a double or blank gameweek). A bench boost gains your bench's points, a
# triple captain one more of your captain's, a wildcard or free hit the better squad it
# buys without hits. Not charged when the chip's set expires within the horizon (the
# first set does after CHIP_SPLIT_GW): it's worth nothing afterwards, so the plan uses
# it in its best remaining week, however small the gain. Higher = save chips for bigger weeks.
CHIP_THRESHOLD_POINTS = {
    "wildcard": 20.0,         # about a season-changing restructure
    "free_hit": 15.0,         # a blank or double gameweek fixes far more than one week
    "bench_boost": 12.0,      # a normal bench expects ~8-10; a double gameweek's ~15-20
    "triple_captain": 10.0,   # a normal premium captain expects ~7-9; a double ~12-15
}

# Event-model weight (Section 3). The expected points the optimiser uses are a blend:
# (1 - w) x the points model (gradient-boosted trees) + w x the event model (team
# ratings -> fixture goals -> each player's goals, assists, clean sheets...). 0 = points
# model only. Chosen on 2025/26 at the half-life above (the lowest RMSE; anything from
# 0.5 to 0.75 was within 0.001), so this season's comparisons are out of sample
# (scripts/study_last_season.py, docs/08).
EVENT_MODEL_WEIGHT = 0.65

# Injured with no return date (Section 3). FPL's chance of playing covers the next
# gameweek only. When the news gives a date ("Expected back 8 Nov", "Suspended until
# 31 Oct") the player counts from the first gameweek after it; when it says "Unknown
# return date" he comes back gradually over the following gameweeks at these chances,
# then fully (so the planner neither writes him off nor counts on him straight away).
UNKNOWN_RETURN_RAMP = (0.25, 0.5, 0.75)

# Accuracy check (the report's Accuracy tab). How many of the most recent finished
# gameweeks to replay and compare with what players actually scored.
ACCURACY_WINDOW_GW = 5

# Opponent archetypes (Section 2). Number of defensive "styles" the 20 teams are
# clustered into, for interpretation and for Claude's narrative.
ARCHETYPE_K = 3

# Minutes floor (Sections 3–4). Players below this predicted start probability
# are treated as bench fodder (not picked to start).
MIN_START_PROB = 0.6

# Bench value (Sections 4–5). Bench players only score when someone in the XI
# doesn't play, so the optimiser counts their expected points at this fraction.
# 0 = pick the cheapest possible bench; higher = a stronger bench.
BENCH_WEIGHT = 0.1

# Claude advisor (Section 6). Reads the week's plan against live football news
# (Anthropic's web search tool) and scores 0-100 how strongly to act on it.
# Needs ANTHROPIC_API_KEY in .env; one request per run with a few searches.
ADVISOR_MODEL = "claude-sonnet-5-5"
ADVISOR_EFFORT = "high"          # low | medium | high | xhigh | max
ADVISOR_MAX_SEARCHES = 6

# Opponent stats source for the opponent_stats adapter. "fpl" derives goals
# conceded per game from the season's own filed FPL results: no browser, and always
# the right season. That is everything FBref's schedule currently gives us, so it is
# the default. "fbref" (soccerdata + Google Chrome) and "understat" (adds PPDA, but
# often Cloudflare-blocked) remain as backends; the adapter falls back on error.
ADVANCED_STATS_SOURCE = "fpl"

# Last season as a prior (Section 2). A player's previous-season averages enter his
# recency-weighted form as if that season ended PREV_SEASON_GAP_GW gameweeks before
# GW1 (the summer break, in decay terms). With half-life 4 and gap 8, last season
# is worth ~1.6 recent matches at GW1, ~0.3 by GW10, almost nothing by GW20. Bigger =
# trust last season less.
PREV_SEASON_GAP_GW = 8.0

# How much a previous-season training row counts relative to this season's when
# the model is fitted (sample_weight), and how much last season's minutes count
# when blending player profiles for the graphics. Two seasons back = this squared.
PREV_SEASON_WEIGHT = 0.5

# Per-90 rates from fewer minutes than this are too noisy to use (left NaN).
PREV_SEASON_MIN_MINUTES = 270

# Player archetypes (graphics + report). Players are clustered per position on
# per-90 profiles: clusters per position, and the minutes needed to be profiled.
# Three per position on purpose: a scatter can only keep 3 colours distinct for
# colour-blind readers, and the report colours each player by type.
PLAYER_ARCHETYPE_K = {"GK": 3, "DEF": 3, "MID": 3, "FWD": 3}
PROFILE_MIN_MINUTES = 450

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
    assert 0 < HORIZON_DECAY <= 1
    assert 0 <= BENCH_WEIGHT < 1
    assert FREE_TRANSFER_VALUE >= 0
    assert CHIP_SETS_PER_SEASON in (1, 2), "chip sets are split at one gameweek (CHIP_SPLIT_GW)"
    assert 1 <= CHIP_SPLIT_GW < GAMEWEEKS_IN_SEASON
    assert set(CHIP_THRESHOLD_POINTS) == set(CHIPS), "one chip threshold per chip"
    assert all(v >= 0 for v in CHIP_THRESHOLD_POINTS.values())
    assert 0 <= EVENT_MODEL_WEIGHT <= 1
    assert all(0 <= x <= 1 for x in UNKNOWN_RETURN_RAMP)
    assert ACCURACY_WINDOW_GW >= 1
    assert PREV_SEASON_GAP_GW >= 0
    assert 0 < PREV_SEASON_WEIGHT <= 1, "a previous season should never outweigh this one"
    assert set(PLAYER_ARCHETYPE_K) == set(POSITION_QUOTA)


if __name__ == "__main__":
    validate()
    print("config OK")
