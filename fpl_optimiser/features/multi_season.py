"""One training table across seasons, with last season weighted in.

Each filed season is built on its own (player and team ids only mean something
within a season), with the previous season as its prior whenever that season is
filed too (prior_season.py, linked by player_link.py). The seasons are then stacked
with:

  season         e.g. '2026-27'
  code           the player's permanent FPL code: the same person in every season
  sample_weight  1 for the latest season, config.PREV_SEASON_WEIGHT for the one
                 before, its square for the one before that... — for the model's
                 fit, so recent seasons count more.
"""
from __future__ import annotations

import pandas as pd

import config
from fpl_optimiser.features import build, opponent, prior_season
from fpl_optimiser.ingest import opponent_stats, player_link, season_store


def build_season(season: str, half_life: float = config.RECENCY_HALF_LIFE_GW) -> pd.DataFrame:
    """The training table for one filed season, with last season as its prior."""
    cur = season_store.load(season)
    prior = None
    prev_season = season_store.previous_season(season)
    if prev_season in season_store.available():
        prev = season_store.load(prev_season)
        links = player_link.link_players(prev.players, cur.players).links
        prior = prior_season.prior_table(prev.players, links,
                                         prev_gameweeks=prev.finished_gameweeks)

    opp = opponent_stats.opponent_stats(season)
    profiles = opponent.assign_archetypes(opponent.build_opponent_profiles(opp, cur.teams))
    tbl = build.build_training_table(cur.history, cur.fixtures, cur.players, profiles,
                                     half_life, prior=prior)
    tbl.insert(0, "season", season)
    tbl.insert(2, "code", tbl["id"].map(cur.players.set_index("id")["code"]))
    return tbl


def build_all(seasons: list[str] | None = None,
              half_life: float = config.RECENCY_HALF_LIFE_GW) -> pd.DataFrame:
    """Every requested (default: every filed) season, stacked, with sample weights."""
    seasons = sorted(seasons or season_store.available())
    latest = int(seasons[-1][:4])
    frames = []
    for season in seasons:
        tbl = build_season(season, half_life)
        tbl["sample_weight"] = config.PREV_SEASON_WEIGHT ** (latest - int(season[:4]))
        frames.append(tbl)
    return pd.concat(frames, ignore_index=True)
