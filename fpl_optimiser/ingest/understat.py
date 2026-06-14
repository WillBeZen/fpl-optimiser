"""Advanced team stats from Understat — the primary advanced-stats source.

Why Understat (and not FBref): FBref now sits behind Cloudflare, so `soccerdata`
drives a headless Chrome browser to read it — heavy, fragile, and it needs Google
Chrome installed. Understat instead embeds its data as JSON inside a <script> tag,
so a single plain HTTP GET is enough (no browser). It also gives exactly the
defensive-style signals we want for opponent characteristics:

  xGA / npxGA   expected goals conceded — how leaky a team is
  ppda          passes allowed per defensive action (THEIR pressing intensity;
                lower = presses harder)
  ppda_allowed  the opponent's PPDA against them (how much they sit off / cede the ball)
  deep_allowed  opponent passes completed near their goal — territory conceded

Player-level data isn't needed here: the FPL API already carries per-player xG/xA
and the new CBIT defensive stats.

Design mirrors fpl_api: a cached HTTP layer (`_cached_get`) and a pure parsing
layer (`parse_team_matches`) that can be unit-tested on a mock page with no network.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd

import config

_CACHE = config.CACHE_DIR / "understat"
_BASE = "https://understat.com/league"
LEAGUE = "EPL"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) fpl-optimiser/0.1"),
}
_DEFAULT_TTL_H = 24.0


def to_understat_season(season: str) -> str:
    """'2025-26' -> '2025'  (Understat uses the season's START year)."""
    return season.split("-")[0]


def _ppda(value) -> float | None:
    """Understat stores ppda as {'att': passes, 'def': defensive_actions}; the
    metric is passes / defensive_actions."""
    if isinstance(value, dict) and value.get("def"):
        return value["att"] / value["def"]
    return None


# --------------------------------------------------------------------------- #
# Parsing layer (pure) — unit-testable on a mock page
# --------------------------------------------------------------------------- #
def _extract_embedded_json(html: str, var_name: str):
    """Understat embeds data as  <var> = JSON.parse('<hex-escaped utf-8>')."""
    m = re.search(var_name + r"\s*=\s*JSON\.parse\('(.*?)'\)", html, re.DOTALL)
    if not m:
        raise ValueError(f"'{var_name}' not found in the Understat page")
    raw = m.group(1)
    # raw is python-style \xHH-escaped UTF-8; this round-trip decodes it correctly
    decoded = raw.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
    return json.loads(decoded)


def parse_team_matches(html: str) -> pd.DataFrame:
    """One row per team per match, with xG/xGA, PPDA and deep completions."""
    teams = _extract_embedded_json(html, "teamsData")  # dict keyed by team id
    rows = []
    for team in teams.values():
        title = team.get("title")
        for h in team.get("history", []):
            rows.append({
                "team": title,
                "date": h.get("date"),
                "home_away": h.get("h_a"),
                "xg": h.get("xG"),
                "xga": h.get("xGA"),
                "npxg": h.get("npxG"),
                "npxga": h.get("npxGA"),
                "ppda": _ppda(h.get("ppda")),
                "ppda_allowed": _ppda(h.get("ppda_allowed")),
                "deep": h.get("deep"),
                "deep_allowed": h.get("deep_allowed"),
                "scored": h.get("scored"),       # goals for
                "missed": h.get("missed"),       # goals conceded
                "xpts": h.get("xpts"),
                "result": h.get("result"),
            })
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# HTTP layer (network) — cached
# --------------------------------------------------------------------------- #
def _cached_get(season_year: str, ttl_hours: float = _DEFAULT_TTL_H) -> str:
    import requests  # lazy

    _CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE / f"{LEAGUE}_{season_year}.html"
    fresh = cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < ttl_hours * 3600
    if fresh:
        return cache_file.read_text(encoding="utf-8")

    url = f"{_BASE}/{LEAGUE}/{season_year}"
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text


# --------------------------------------------------------------------------- #
# Convenience: fetch + parse (network)
# --------------------------------------------------------------------------- #
def team_matches(season: str = config.SEASON_TRAIN) -> pd.DataFrame:
    """Per-team per-match advanced stats for the season."""
    return parse_team_matches(_cached_get(to_understat_season(season)))


def team_season(season: str = config.SEASON_TRAIN) -> pd.DataFrame:
    """Season aggregates per team (the raw material for opponent characteristics).

    These are plain means/sums; recency weighting is applied in Section 2.
    """
    m = team_matches(season)
    agg = (m.groupby("team")
             .agg(matches=("date", "count"),
                  xga_mean=("xga", "mean"),
                  npxga_mean=("npxga", "mean"),
                  deep_allowed_mean=("deep_allowed", "mean"),
                  ppda_mean=("ppda", "mean"),
                  ppda_allowed_mean=("ppda_allowed", "mean"),
                  goals_conceded=("missed", "sum"),
                  goals_scored=("scored", "sum"))
             .reset_index())
    return agg
