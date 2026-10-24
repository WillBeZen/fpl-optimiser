"""Season store — FPL data filed by season, so seasons never overwrite each other.

The live FPL API only serves the CURRENT season, and it re-assigns player and team
ids every season. So each season's raw JSON is filed into its own folder:

    data/raw/fpl/<season>/bootstrap-static.json
                         /fixtures.json
                         /element-summary_<id>.json   one per player
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
from dataclasses import dataclass
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
    history: pd.DataFrame      # per-match rows for every filed player, with `id`
    past: pd.DataFrame         # element-summary history_past: earlier seasons' totals, with `id`

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
    fetched = 0
    for i, pid in enumerate(ids, 1):
        f = d / f"element-summary_{pid}.json"
        if not _fresh(f, ttl_hours):
            _write(f, fpl_api.fetch(f"element-summary/{pid}"))
            fetched += 1
            time.sleep(pause_s)
        if i % 100 == 0 or i == len(ids):
            log(f"   {i}/{len(ids)} players  ({fetched} fetched, {i - fetched} already fresh)")
    return season


# --------------------------------------------------------------------------- #
# Reading a filed season back
# --------------------------------------------------------------------------- #
def assemble(season: str, bootstrap: dict, fixtures: list,
             summaries: dict[int, dict]) -> Season:
    """Pure: raw payloads -> a Season of tidy tables (`summaries` keyed by player id)."""
    teams = fpl_api.parse_teams(bootstrap)

    def stack(key: str) -> pd.DataFrame:
        frames = [pd.DataFrame(s.get(key, [])).assign(id=pid)
                  for pid, s in sorted(summaries.items()) if s.get(key)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return Season(season=season, bootstrap=bootstrap,
                  players=fpl_api.parse_players(bootstrap), teams=teams,
                  fixtures=fpl_api.parse_fixtures(fixtures, teams),
                  gameweeks=fpl_api.parse_gameweeks(bootstrap),
                  history=stack("history"), past=stack("history_past"))


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
    return assemble(season, bootstrap, json.loads((d / "fixtures.json").read_text()), summaries)
