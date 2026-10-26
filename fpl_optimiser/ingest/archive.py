"""Earlier seasons' match records from a public archive (FPL's API only serves the
current season).

2025/26 was filed from FPL's API late in the season: every player's season totals,
but match-by-match records only for the top 50 (season_store.py). The community
archive vaastav/Fantasy-Premier-League kept every player's match records all season.
Its `merged_gw.csv` holds FPL's own element-summary history rows, plus the player's
name, position and club.

    data/raw/archive/<season>/merged_gw.csv     downloaded once, pinned to a commit

- `pull(season)` downloads the file listed in SOURCES and checks its git blob id,
  so it is exactly the file that was checked (network, once per season).
- `parse_merged_gw(df)` (pure) keeps FPL's history fields, with `id` = `element`
  (that season's FPL id), and drops exact repeats (the 2025/26 file repeats 10 rows).
- `load(season)` reads it back (None if it isn't downloaded).
- `checks(...)` compares it with FPL's own data: the match files FPL served, its
  season totals and the fixture list.

season_store.load() uses it for every player FPL's own files don't cover; FPL's
filed folder is never modified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config

ARCHIVE = config.RAW_DIR / "archive"
FILE = "merged_gw.csv"
SOURCES = {
    "2025-26": {
        "url": ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
                "f9ed3e8839b0f970e0d5d4a83c5628f6eaee755a/data/2025-26/gws/merged_gw.csv"),
        "commit": "f9ed3e8 'Final 2025-26 update' (17 June 2026)",
        "blob": "b2a5ec06d398224e325991016dff5770b249c881",      # git's id for the file's content
        "bytes": 5_445_456,
    },
}
EXTRA = ("name", "position", "team", "xP", "GW")     # the archive's own columns, not FPL's
TOTALS = ["total_points", "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
          "saves", "bonus", "bps", "expected_goals", "expected_assists", "expected_goal_involvements",
          "expected_goals_conceded", "defensive_contribution", "clearances_blocks_interceptions",
          "recoveries", "tackles"]
COMPARED = ["round", "opponent_team", "was_home", "minutes", "total_points", "goals_scored", "assists",
            "clean_sheets", "goals_conceded", "saves", "bonus", "bps", "starts", "expected_goals",
            "expected_assists", "expected_goals_conceded", "defensive_contribution", "value"]


def path(season: str, root: Path | None = None) -> Path:
    return (ARCHIVE if root is None else root) / season / FILE


def blob_id(data: bytes) -> str:
    """Git's id for a file's content, as GitHub lists it: sha1 of 'blob <size>\\0' + data."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def pull(season: str, root: Path | None = None) -> Path:
    """Download the season's archive file (unless it is already there) and check that
    it is the pinned one before keeping it."""
    from fpl_optimiser.ingest import fpl_api

    src = SOURCES[season]
    p = path(season, root)
    if p.exists() and blob_id(p.read_bytes()) == src["blob"]:
        return p
    resp = fpl_api._http().get(src["url"], timeout=60)
    resp.raise_for_status()
    if blob_id(resp.content) != src["blob"]:
        raise ValueError(f"{season}: the downloaded file isn't the one pinned in archive.SOURCES")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(p)
    return p


def parse_merged_gw(df: pd.DataFrame) -> pd.DataFrame:
    """Pure: the archive's rows as season_store history rows (`id` = that season's
    FPL id), exact repeats dropped (how many: `.attrs["repeats"]`)."""
    rows = df.drop(columns=[c for c in EXTRA if c in df])
    out = rows.drop_duplicates().reset_index(drop=True)
    out["id"] = out["element"].astype(int)
    out.attrs["repeats"] = len(rows) - len(out)
    return out


def load(season: str, root: Path | None = None) -> pd.DataFrame | None:
    """The season's archive rows, or None if the file isn't downloaded."""
    p = path(season, root)
    return parse_merged_gw(pd.read_csv(p, low_memory=False)) if p.exists() else None


# --------------------------------------------------------------------------- #
# Data checks: the archive against FPL's own data
# --------------------------------------------------------------------------- #
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def checks(arc: pd.DataFrame, fpl_history: pd.DataFrame, players: pd.DataFrame,
           fixtures: pd.DataFrame, season: str = "") -> list[dict]:
    """[{check, ok, detail}] (as accuracy.reconcile): the archive's rows against the
    fixture list, against the match files FPL served (`fpl_history`) and against
    FPL's season totals (`players`)."""
    out = []
    label = f"{season} archive" if season else "The archive"

    def add(check, ok, detail):
        out.append({"check": f"{label}: {check}", "ok": bool(ok), "detail": detail})

    fx = fixtures.set_index("fixture_id")
    known = arc["fixture"].isin(fx.index)
    a = arc[known]
    opp = np.where(a["was_home"].astype(bool), a["fixture"].map(fx["away_id"]), a["fixture"].map(fx["home_id"]))
    same = ((a["opponent_team"] == opp) & (a["round"] == a["fixture"].map(fx["gameweek"]))
            & (_num(a["team_h_score"]) == a["fixture"].map(fx["home_score"]))
            & (_num(a["team_a_score"]) == a["fixture"].map(fx["away_score"])))
    add("matches the fixture list", bool(known.all() and same.all()),
        f"{int(same.sum())} of {len(arc)} match records agree with FPL's fixtures (gameweek, opponent, "
        f"home or away, score)" + (f"; {int((~known).sum())} name an unknown fixture" if not known.all() else ""))

    ours = set(fpl_history["id"]) if len(fpl_history) else set()
    both = (fpl_history.merge(arc[arc["id"].isin(ours)], on=["id", "fixture"], how="outer",
                              suffixes=("_fpl", "_arc"), indicator=True) if ours else pd.DataFrame())
    if ours:
        m = both[both["_merge"] == "both"]
        differ = sum(int((~np.isclose(_num(m[f"{c}_fpl"]).fillna(-1), _num(m[f"{c}_arc"]).fillna(-1),
                                      atol=1e-6)).sum()) for c in COMPARED if f"{c}_fpl" in m and f"{c}_arc" in m)
        extra = int((both["_merge"] != "both").sum())
        add("agrees with FPL's own match files", differ == 0 and extra == 0,
            f"{len(m)} match records of the {len(ours)} players FPL's files cover: "
            + ("identical" if differ == 0 and extra == 0 else f"{differ} values differ, {extra} records in one only"))

    tot = players.set_index("id")
    sums = arc.groupby("id")[[c for c in TOTALS if c in arc]].sum(numeric_only=True)
    cols = [c for c in TOTALS if c in tot and c in sums]
    off = pd.Series(False, index=tot.index)
    for c in cols:
        tol = 0.05 if c.startswith("expected") else 1e-9
        diff = (sums[c].reindex(tot.index).fillna(0) - _num(tot[c]).fillna(0)).abs()
        off |= diff > tol
    unknown = set(sums.index) - set(tot.index)
    add("adds up to FPL's season totals", not off.any() and not unknown,
        f"{int((~off).sum())} of {len(tot)} players' match records sum to FPL's season totals "
        f"({len(cols)} stats: points, minutes, goals, assists, bonus, xG, defensive contributions, ...)"
        + (f"; {len(unknown)} players FPL doesn't list" if unknown else "")
        + (f"; repeated rows dropped first: {arc.attrs.get('repeats', 0)}" if arc.attrs.get("repeats") else ""))
    return out


def season_checks(season: str, root: Path | None = None, store: Path | None = None) -> list[dict]:
    """checks() for a filed season with an archive file ([] without one)."""
    from fpl_optimiser.ingest import season_store

    arc = load(season, root)
    if arc is None:
        return []
    fpl = season_store.load(season, store, archive=False)
    return checks(arc, fpl.history, fpl.players, fpl.fixtures, season)
