"""FPL API client — fetch and tidy the public Fantasy Premier League data.

Design: the HTTP layer (`_cached_get`) and the parsing layer (`parse_*`) are kept
separate, so the parsing can be unit-tested on mock payloads with no network.

Live calls hit https://fantasy.premierleague.com/api (no auth). Responses are
cached to data/cache/fpl/ with a short TTL so repeated runs in a session don't
re-hit the API. `requests` is imported lazily so this module loads without it.
"""
from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd

import config

_CACHE = config.CACHE_DIR / "fpl"
_HEADERS = {"User-Agent": "fpl-optimiser/0.1 (personal project)"}
_DEFAULT_TTL_H = 6.0

# FPL element_type id -> our position code (FPL uses "GKP"; we use "GK").
_POSITION_BY_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Player stat columns we lift from the API. Any that are absent get filled with
# NA, so a schema tweak on FPL's side never crashes ingestion.
_PLAYER_STAT_COLS = [
    "total_points", "minutes", "starts",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
    "bonus", "bps", "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    # 2025/26 defensive-contribution inputs (CBIT / CBIRT):
    "clearances_blocks_interceptions", "recoveries", "tackles",
    "defensive_contribution",
    # market / availability:
    "selected_by_percent", "form", "now_cost",
    "chance_of_playing_next_round", "status",
]

_NUMERIC_COLS = [
    "selected_by_percent", "form", "ict_index", "influence", "creativity",
    "threat", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
]


def _series(df: pd.DataFrame, col: str, default=pd.NA) -> pd.Series:
    """Return df[col] if present, else an NA-filled series (resilient parsing)."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


# --------------------------------------------------------------------------- #
# HTTP layer (network) — thin, cached
# --------------------------------------------------------------------------- #
def _cached_get(path: str, ttl_hours: float = _DEFAULT_TTL_H) -> Any:
    """GET {FPL_API_BASE}/{path}/ with a small on-disk JSON cache."""
    import requests  # lazy: module still imports if requests isn't installed

    _CACHE.mkdir(parents=True, exist_ok=True)
    key = path.strip("/").replace("/", "_") or "root"
    cache_file = _CACHE / f"{key}.json"

    fresh = cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < ttl_hours * 3600
    if fresh:
        return json.loads(cache_file.read_text())

    url = f"{config.FPL_API_BASE}/{path.strip('/')}/"
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    cache_file.write_text(json.dumps(data))
    return data


# --------------------------------------------------------------------------- #
# Parsing layer (pure) — unit-testable on mock dicts
# --------------------------------------------------------------------------- #
def parse_teams(bootstrap: dict) -> pd.DataFrame:
    teams = pd.DataFrame(bootstrap["teams"])
    cols = ["id", "name", "short_name", "strength",
            "strength_attack_home", "strength_attack_away",
            "strength_defence_home", "strength_defence_away"]
    cols = [c for c in cols if c in teams.columns]
    return teams[cols].copy()


def parse_players(bootstrap: dict) -> pd.DataFrame:
    players = pd.DataFrame(bootstrap["elements"])
    teams = parse_teams(bootstrap).set_index("id")

    keep = ["id", "web_name", "first_name", "second_name", "team", "element_type"]
    for c in _PLAYER_STAT_COLS:
        if c not in players.columns:
            players[c] = pd.NA
    df = players[keep + _PLAYER_STAT_COLS].copy()

    df["position"] = df["element_type"].map(_POSITION_BY_TYPE)
    df["team_name"] = df["team"].map(teams["name"])
    df["team_short"] = df["team"].map(teams["short_name"])
    df["price"] = pd.to_numeric(df["now_cost"], errors="coerce") / 10.0  # tenths -> £m

    for c in _NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.drop(columns=["now_cost"])


def parse_gameweeks(bootstrap: dict) -> pd.DataFrame:
    ev = pd.DataFrame(bootstrap["events"])
    cols = ["id", "name", "deadline_time", "finished", "data_checked",
            "is_previous", "is_current", "is_next",
            "average_entry_score", "highest_score"]
    cols = [c for c in cols if c in ev.columns]
    out = ev[cols].copy()
    if "deadline_time" in out.columns:
        out["deadline_time"] = pd.to_datetime(out["deadline_time"], errors="coerce", utc=True)
    return out


def parse_fixtures(fixtures: list, teams: pd.DataFrame) -> pd.DataFrame:
    fx = pd.DataFrame(fixtures)
    name = teams.set_index("id")["name"]
    out = pd.DataFrame({
        "fixture_id": fx["id"],
        "gameweek": fx["event"],
        "kickoff_time": pd.to_datetime(_series(fx, "kickoff_time"), errors="coerce", utc=True),
        "finished": _series(fx, "finished"),
        "home_id": fx["team_h"],
        "away_id": fx["team_a"],
        "home": fx["team_h"].map(name),
        "away": fx["team_a"].map(name),
        "home_score": _series(fx, "team_h_score"),
        "away_score": _series(fx, "team_a_score"),
        "home_fdr": _series(fx, "team_h_difficulty"),
        "away_fdr": _series(fx, "team_a_difficulty"),
    })
    return out


def parse_player_history(element_summary: dict) -> pd.DataFrame:
    """Per-match rows for a single player this season (from element-summary)."""
    return pd.DataFrame(element_summary.get("history", []))


# --------------------------------------------------------------------------- #
# Convenience: fetch + parse (network)
# --------------------------------------------------------------------------- #
def players() -> pd.DataFrame:
    return parse_players(_cached_get("bootstrap-static"))


def teams() -> pd.DataFrame:
    return parse_teams(_cached_get("bootstrap-static"))


def gameweeks() -> pd.DataFrame:
    return parse_gameweeks(_cached_get("bootstrap-static"))


def fixtures() -> pd.DataFrame:
    bs = _cached_get("bootstrap-static")
    return parse_fixtures(_cached_get("fixtures"), parse_teams(bs))


def player_history(player_id: int) -> pd.DataFrame:
    return parse_player_history(_cached_get(f"element-summary/{player_id}", ttl_hours=12))


def current_gameweek() -> int | None:
    gw = gameweeks()
    for flag in ("is_current", "is_next"):
        if flag in gw.columns:
            hit = gw.loc[gw[flag] == True, "id"]  # noqa: E712 (pandas truthiness)
            if len(hit):
                return int(hit.iloc[0])
    return None


def my_squad(manager_id: int, gameweek: int) -> dict:
    """Your picks + bank + squad value for a gameweek (entry/.../picks).

    Note: the public API does not expose 'free transfers available this week';
    that has to be inferred from transfer history (handled in Section 5) or set
    by you. Here we return what the endpoint gives directly.
    """
    raw = _cached_get(f"entry/{manager_id}/event/{gameweek}/picks", ttl_hours=1)
    picks = pd.DataFrame(raw["picks"])  # element, position, multiplier, is_captain, is_vice_captain
    eh = raw.get("entry_history", {})
    return {
        "picks": picks,
        "active_chip": raw.get("active_chip"),
        "bank": eh.get("bank", 0) / 10.0,          # tenths -> £m
        "squad_value": eh.get("value", 0) / 10.0,
        "event_transfers": eh.get("event_transfers"),
        "event_transfers_cost": eh.get("event_transfers_cost"),
    }
