"""OPTIONAL advanced team stats from FBref via `soccerdata`.

⚠️  Secondary source. FBref is behind Cloudflare, so `soccerdata` drives a
headless Chrome browser to read it — you must have Google Chrome installed
(`brew install --cask google-chrome` on macOS). The PRIMARY advanced-stats
source is `understat.py`, which needs no browser; use this module only if you
specifically want FBref's detailed defensive-action tables.

For v1 the opponent-characteristics model is TEAM-level, so this module pulls
team season + match tables. Player advanced stats are not needed yet — the FPL
API already carries per-player xG/xA and the new CBIT defensive stats.

`soccerdata` caches every download to disk and rate-limits FBref for you; do NOT
disable that. The cache location is set via the SOCCERDATA_DIR env var below
(confirm it for your installed soccerdata version; otherwise it falls back to the
soccerdata default, which is harmless).

`soccerdata` is imported lazily inside each function, so this module loads even
when soccerdata isn't installed yet.

What feeds the defensive profile (built in Section 2):
  - schedule()            -> per-match home/away xG  => each team's xG CONCEDED
  - 'keeper'  table       -> goals against, save %, clean sheets
  - 'defense' table       -> tackles / interceptions / blocks / clearances (style)
  - 'misc'    table       -> aerials won, fouls, cards, recoveries
  - 'possession' table    -> possession %, territory
Note: a true PPDA / pressing metric is not reliably available from FBref, so we
use the proxies above. Exact FBref column names vary slightly by season —
inspect once on first run (see docs/01_data_ingestion.md).
"""
from __future__ import annotations

import os

import pandas as pd

import config

# Point soccerdata's cache at our data/cache (best-effort; see module docstring).
os.environ.setdefault("SOCCERDATA_DIR", str(config.CACHE_DIR / "soccerdata"))

FBREF_LEAGUE = "ENG-Premier League"

# Season-level stat tables used to characterise each team's defensive profile.
TEAM_STAT_TYPES = ("standard", "shooting", "defense", "possession", "misc", "keeper")


def to_soccerdata_season(season: str) -> str:
    """'2025-26' -> '2526' (soccerdata's compact season code).

    soccerdata accepts several forms and will raise an error listing the valid
    options if this code is wrong for your version.
    """
    start, end = season.split("-")
    return start[-2:] + end[-2:]


def _fbref(season: str):
    import soccerdata as sd
    return sd.FBref(leagues=FBREF_LEAGUE, seasons=to_soccerdata_season(season))


def schedule(season: str = config.SEASON_TRAIN) -> pd.DataFrame:
    """Match list with home/away xG — the light source for per-match xG for/against."""
    return _fbref(season).read_schedule()


def team_season_stats(season: str = config.SEASON_TRAIN,
                      stat_types: tuple[str, ...] = TEAM_STAT_TYPES) -> dict[str, pd.DataFrame]:
    """{stat_type: DataFrame} of team season tables (a team's own season stats)."""
    fb = _fbref(season)
    return {st: fb.read_team_season_stats(stat_type=st) for st in stat_types}


def team_match_stats(season: str = config.SEASON_TRAIN,
                     stat_type: str = "defense") -> pd.DataFrame:
    """Per-team per-match table for one stat type.

    Heavier than the season tables (more requests). Used in Section 2 to build
    recency-weighted rolling defensive profiles.
    """
    return _fbref(season).read_team_match_stats(stat_type=stat_type)
