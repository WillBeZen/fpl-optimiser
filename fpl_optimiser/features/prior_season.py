"""Last season as a prior for this season — leakage-free by construction.

A player's 2025/26 record reaches his 2026/27 rows through the cross-season link
table (ingest/player_link.py). Last season is entirely in the past, so nothing here
can leak future information into a 2026/27 row.

`prior_table()` gives, per player (keyed by THIS season's id):

  pg_<stat>        last season's per-gameweek average of each stat the form features
                   average (points, minutes, xG, xA, xGI, defensive contributions,
                   bonus, BPS), plus pg_played_60 (starts per gameweek, standing in
                   for the 60-minute rate). build.py blends these into each row's
                   recency-weighted form as if last season ended
                   config.PREV_SEASON_GAP_GW gameweeks before GW1.
  prev_*           explicit context for the model: points and minutes per gameweek,
                   points / xGI / defensive contributions per 90, and whether he has
                   changed club since.
  has_prev_season  True (players without a previous-season record get no row; the
                   table builder marks them False).

Approximation: per-gameweek averages divide season totals by the season's
gameweeks, so a January arrival looks like a part-timer until this season's matches
(which soon outweigh the prior) correct it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from fpl_optimiser.features import build


def prior_table(prev_players: pd.DataFrame, links: pd.DataFrame,
                prev_gameweeks: int = config.GAMEWEEKS_IN_SEASON,
                min_minutes: int = config.PREV_SEASON_MIN_MINUTES) -> pd.DataFrame:
    """Last season, per linked player, keyed by this season's `id`.

    prev_players : last season's fpl_api.parse_players table (season totals)
    links        : player_link.link_players(prev, curr).links
    """
    totals = prev_players.set_index("id").reindex(links["id_prev"])

    def stat(col: str) -> np.ndarray:
        if col not in totals:
            return np.full(len(totals), np.nan)
        return pd.to_numeric(totals[col], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame({"id": links["id_curr"].to_numpy()})
    for col in build._FORM_FIELDS.values():
        out[f"pg_{col}"] = stat(col) / prev_gameweeks
    out["pg_played_60"] = stat("starts") / prev_gameweeks

    minutes = stat("minutes")
    enough = np.where(minutes >= min_minutes, minutes, np.nan)     # too few -> NaN rate
    out["has_prev_season"] = True
    out["prev_points_per_gw"] = out["pg_total_points"]
    out["prev_minutes_per_gw"] = out["pg_minutes"]
    out["prev_points_per_90"] = stat("total_points") / enough * 90
    out["prev_xgi_per_90"] = stat("expected_goal_involvements") / enough * 90
    out["prev_defcon_per_90"] = stat("defensive_contribution") / enough * 90
    out["prev_changed_team"] = links["changed_team"].to_numpy(dtype=bool)
    return out
