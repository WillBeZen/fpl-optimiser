"""Season store — FPL data filed by season, so seasons never overwrite each other.

The live FPL API only serves the CURRENT season, and it re-assigns player and team
ids every season. So each season's raw JSON is filed into its own folder:

    data/raw/fpl/<season>/bootstrap-static.json
                         /fixtures.json
                         /element-summary_<id>.json   one per player
                         /event-live_<gw>.json        FPL's official points per
                                                      finished gameweek (filed once)
                         /FROZEN                      present => read-only

- `pull_live()` files the live season. The season is read from the data itself
  (GW1's deadline), never assumed. It is resumable: player files younger than
  `ttl_hours` are kept rather than re-fetched.
- `freeze()` marks a finished season read-only; `pull_live()` refuses to touch it.
- `load()` reads a filed season back as tidy tables for the feature code.

Players are linked ACROSS seasons by their permanent `code` (see player_link.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
from fpl_optimiser.ingest import fpl_api

STORE = config.RAW_DIR / "fpl"
_FROZEN = "FROZEN"


@dataclass
class Season:
    """One season's FPL data. Ids in every table are THIS season's ids."""
    season: str
    bootstrap: dict
    players: pd.DataFrame      # fpl_api.parse_players
    teams: pd.DataFrame        # fpl_api.parse_teams
    fixtures: pd.DataFrame     # fpl_api.parse_fixtures
    gameweeks: pd.DataFrame    # fpl_api.parse_gameweeks
    history: pd.DataFrame      # per-match rows for every filed player, with `id` (finished matches)
    past: pd.DataFrame         # element-summary history_past: earlier seasons' totals, with `id`
    pending: pd.DataFrame = field(default_factory=pd.DataFrame)   # match rows not finished yet
    live: pd.DataFrame = field(default_factory=pd.DataFrame)      # official points per finished GW

    @property
    def finished_gameweeks(self) -> int:
        return int(self.gameweeks["finished"].fillna(False).astype(bool).sum())


def season_of(bootstrap: dict) -> str:
    """'2026-27' — from the earliest gameweek deadline (August of the start year)."""
    first = min(e["deadline_time"] for e in bootstrap["events"])
    year = int(first[:4])
    return f"{year}-{(year + 1) % 100:02d}"


def season_dir(season: str, root: Path | None = None) -> Path:
    """Where a season is filed. `root` defaults to STORE (read at call time, so
    tests can point the whole store at a temporary folder)."""
    return (STORE if root is None else root) / season


def is_frozen(season: str, root: Path | None = None) -> bool:
    return (season_dir(season, root) / _FROZEN).exists()


def freeze(season: str, note: str = "", root: Path | None = None) -> None:
    """Mark a filed season as complete and read-only."""
    d = season_dir(season, root)
    if not (d / "bootstrap-static.json").exists():
        raise FileNotFoundError(f"nothing filed for {season} in {d}")
    (d / _FROZEN).write_text(note or f"{season} is complete; pull_live() will not modify it.\n")


def available(root: Path | None = None) -> list[str]:
    """Seasons that have been filed, oldest first."""
    root = STORE if root is None else root
    return sorted(p.name for p in root.glob("*-*") if (p / "bootstrap-static.json").exists())


# --------------------------------------------------------------------------- #
# Filing the live season (network)
# --------------------------------------------------------------------------- #
def _write(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")        # write-then-rename: never a half-written file
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _fresh(path: Path, ttl_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600


MATCH_HOURS = 2.5          # a match is over (and its data out) this long after kick-off


def last_played(fixtures: list, now: float | None = None) -> dict[int, float]:
    """Club id -> when its latest match ended (epoch seconds, kick-off + MATCH_HOURS),
    counting only matches already over."""
    now = time.time() if now is None else now
    out: dict[int, float] = {}
    for fx in fixtures:
        kick = fx.get("kickoff_time")
        if not kick:
            continue
        end = pd.Timestamp(kick).timestamp() + MATCH_HOURS * 3600
        if end <= now:
            for team in (fx.get("team_h"), fx.get("team_a")):
                out[team] = max(out.get(team, 0.0), end)
    return out


def pull_live(limit: int | None = None, ttl_hours: float = 12.0, pause_s: float = 0.1,
              root: Path | None = None, log=print) -> str:
    """File the live season (bootstrap, fixtures, every player's element-summary).

    `limit` keeps only the top-N players by points (quick runs). `pause_s` spaces
    out the ~700 player requests to stay polite to FPL. Returns the season filed.
    """
    bootstrap = fpl_api.fetch("bootstrap-static")
    season = season_of(bootstrap)
    if is_frozen(season, root):
        raise PermissionError(f"{season} is frozen ({season_dir(season, root)}); not overwriting it")
    d = season_dir(season, root)
    d.mkdir(parents=True, exist_ok=True)
    _write(d / "bootstrap-static.json", bootstrap)
    _write(d / "fixtures.json", fpl_api.fetch("fixtures"))

    ranked = sorted(bootstrap["elements"], key=lambda p: p.get("total_points") or 0, reverse=True)
    ids = [p["id"] for p in ranked][:limit]
    team_of = {p["id"]: p.get("team") for p in bootstrap["elements"]}
    played = last_played(json.loads((d / "fixtures.json").read_text()))
    fetched = 0
    for i, pid in enumerate(ids, 1):
        f = d / f"element-summary_{pid}.json"
        # re-fetch if older than ttl_hours, or if his club has played since it was fetched
        # (otherwise FPL's live season totals run ahead of the filed match records)
        stale = f.exists() and played.get(team_of.get(pid), 0) > f.stat().st_mtime
        if stale or not _fresh(f, ttl_hours):
            _write(f, fpl_api.fetch(f"element-summary/{pid}"))
            fetched += 1
            time.sleep(pause_s)
        if i % 100 == 0 or i == len(ids):
            log(f"   {i}/{len(ids)} players  ({fetched} fetched, {i - fetched} already fresh)")

    # each finished gameweek's official points: final once FPL has checked them, so
    # filed once and never re-fetched
    for e in bootstrap.get("events", []):
        if e.get("finished") and e.get("data_checked", True):
            f = d / f"event-live_{e['id']}.json"
            if not f.exists():
                _write(f, fpl_api.fetch(f"event/{e['id']}/live"))
                time.sleep(pause_s)
    return season


# --------------------------------------------------------------------------- #
# Reading a filed season back
# --------------------------------------------------------------------------- #
def assemble(season: str, bootstrap: dict, fixtures: list,
             summaries: dict[int, dict], live: dict[int, dict] | None = None) -> Season:
    """Pure: raw payloads -> a Season of tidy tables (`summaries` keyed by player id,
    `live` by gameweek)."""
    teams = fpl_api.parse_teams(bootstrap)

    def stack(key: str) -> pd.DataFrame:
        frames = [pd.DataFrame(s.get(key, [])).assign(id=pid)
                  for pid, s in sorted(summaries.items()) if s.get(key)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    parsed = fpl_api.parse_fixtures(fixtures, teams)
    played, pending = split_played(stack("history"), parsed)
    frames = [fpl_api.parse_event_live(p, gw) for gw, p in sorted((live or {}).items())]
    return Season(season=season, bootstrap=bootstrap,
                  players=fpl_api.parse_players(bootstrap), teams=teams, fixtures=parsed,
                  gameweeks=fpl_api.parse_gameweeks(bootstrap),
                  history=played, past=stack("history_past"), pending=pending,
                  live=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def played_only(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Only match records whose fixture FPL has marked finished.

    Once a gameweek starts, FPL lists every player's match in it, including those
    not yet kicked off (0 minutes, 0 points) and those still being played. Counted
    as real games they would drag form down and teach the models that players
    didn't play, so they are dropped until the fixture is finished."""
    return split_played(history, fixtures)[0]


def split_played(history: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(finished matches, matches not finished yet): see played_only. Together they
    are every match record FPL listed, so their points add up to its season totals."""
    if history.empty or "fixture" not in history or "finished" not in fixtures:
        return history, history.iloc[0:0]
    done = history["fixture"].isin(set(fixtures.loc[fixtures["finished"] == True, "fixture_id"]))  # noqa: E712
    return history[done].reset_index(drop=True), history[~done].reset_index(drop=True)


def previous_season(season: str) -> str:
    """'2026-27' -> '2025-26'."""
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def load_fixtures(season: str, root: Path | None = None) -> pd.DataFrame:
    """Just a filed season's parsed fixtures (cheap: skips the player files)."""
    d = season_dir(season, root)
    bootstrap = json.loads((d / "bootstrap-static.json").read_text())
    return fpl_api.parse_fixtures(json.loads((d / "fixtures.json").read_text()),
                                  fpl_api.parse_teams(bootstrap))


def load(season: str, root: Path | None = None) -> Season:
    """Read a filed season. Refuses a folder whose data is from another season."""
    d = season_dir(season, root)
    bootstrap = json.loads((d / "bootstrap-static.json").read_text())
    if season_of(bootstrap) != season:
        raise ValueError(f"{d} holds {season_of(bootstrap)} data, not {season}")
    summaries = {int(f.stem.split("_")[1]): json.loads(f.read_text())
                 for f in d.glob("element-summary_*.json")}
    live = {int(f.stem.split("_")[1]): json.loads(f.read_text()) for f in d.glob("event-live_*.json")}
    return assemble(season, bootstrap, json.loads((d / "fixtures.json").read_text()), summaries, live)
