"""Assemble the model training table: one row per (player, gameweek) played.

Each row carries:
  - targets:   target_points (what Section 3 predicts) and minutes (for the minutes model)
  - player form: recency-weighted aggregates over the player's STRICTLY-PRIOR matches
                 (no leakage — round t only ever sees rounds < t)
  - opponent:  the opponent faced that week — as-of rolling goals conceded (leakage-free),
               plus the season profile's archetype and FPL defensive strength
  - interaction: attacking form x opponent leakiness
  - cold-start: a flag + position-median fallback for players with little history

How it stays fast: `weighted_form_asof` is the readable definition of one row's
form. The table builder computes exactly the same numbers for every row at once:
per player it builds one small "as-of" weight matrix (`_asof_weights`) whose entry
[i, j] is the weight match j gets when predicting match i — zero unless j was in a
strictly earlier round — so all form features become a couple of matrix products.
A full season (~30k rows) builds in well under a second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from fpl_optimiser.features import recency

# player per-match stats → recency-weighted "form" features (per game)
_FORM_FIELDS = {
    "form_points_pg": "total_points",
    "form_minutes_pg": "minutes",
    "form_xg_pg": "expected_goals",
    "form_xa_pg": "expected_assists",
    "form_xgi_pg": "expected_goal_involvements",
    "form_defcon_pg": "defensive_contribution",
    "form_bonus_pg": "bonus",
    "form_bps_pg": "bps",
}

_OUTPUT_COLUMNS = [
    "id", "round", "web_name", "position", "was_home", "opponent_team",
    "target_points", "minutes",
    *_FORM_FIELDS, "games_played", "form_starts_rate",
    "opp_gc_pg_asof", "opp_archetype", "opp_str_def", "x_xgi_oppgc", "is_cold",
]


def weighted_form_asof(player_history: pd.DataFrame, as_of_round: int,
                       half_life: float = config.RECENCY_HALF_LIFE_GW) -> dict:
    """Recency-weighted form from rounds strictly BEFORE `as_of_round` (no leakage).

    This is the one-row reference definition; `build_training_table` computes the
    same values for every row at once (and the tests check they agree).
    """
    h = player_history[player_history["round"] < as_of_round]
    if h.empty:
        out = {k: float("nan") for k in _FORM_FIELDS}
        out.update(games_played=0, form_starts_rate=float("nan"))
        return out

    g_ago = (as_of_round - h["round"]).to_numpy()
    out: dict = {}
    for feat, col in _FORM_FIELDS.items():
        if col in h.columns:
            vals = pd.to_numeric(h[col], errors="coerce").to_numpy()
        else:
            vals = np.full(len(h), np.nan)
        out[feat] = recency.weighted_mean(vals, g_ago, half_life)

    mins = pd.to_numeric(h["minutes"], errors="coerce").to_numpy() if "minutes" in h else np.full(len(h), np.nan)
    out["games_played"] = int(np.nansum(mins > 0))
    out["form_starts_rate"] = recency.weighted_mean((mins >= 60).astype(float), g_ago, half_life)
    return out


# --------------------------------------------------------------------------- #
# Vectorised as-of helpers (same maths as weighted_form_asof, all rows at once)
# --------------------------------------------------------------------------- #
def _asof_weights(rounds: np.ndarray,
                  half_life: float) -> tuple[np.ndarray, np.ndarray]:
    """As-of weights for one player's (or team's) matches.

    W[i, j] is the weight match j gets when building row i's features:
    0.5 ** (gameweeks between them / half_life) if match j was in a STRICTLY
    earlier round, else 0 — so a row never sees its own round or later.
    prior[i, j] is 1 where match j counts for row i, else 0.
    """
    gap = rounds[:, None] - rounds[None, :]          # gameweeks from match j to row i
    prior = (gap > 0).astype(float)
    W = recency.decay_weight(np.clip(gap, 0, None), half_life) * prior
    return W, prior


def _weighted_means(W: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`recency.weighted_mean` for every row (of W) and column (of values) at once:
    sum(w * v) / sum(w) over the non-NaN values; NaN where nothing is usable."""
    usable = ~np.isnan(values)
    num = W @ np.where(usable, values, 0.0)
    den = W @ usable.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """df[col] as numbers (FPL sends some as strings, e.g. expected_goals);
    all-NaN if the column is absent (resilient parsing)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _floats(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype=float, na_value=np.nan)


def team_defence_asof(fixtures: pd.DataFrame,
                      half_life: float = config.RECENCY_HALF_LIFE_GW) -> pd.DataFrame:
    """(team_id, round) → recency-weighted goals conceded over that team's PRIOR rounds.

    One row per unique (team, round): in a double gameweek both matches share one
    value, built only from earlier rounds. Empty (but well-formed) before any
    fixture has finished.
    """
    fx = fixtures.dropna(subset=["gameweek", "home_score", "away_score"])
    if "finished" in fx.columns:
        fx = fx[fx["finished"] == True]  # noqa: E712

    # each fixture -> two rows: a team concedes the OTHER side's goals
    rnd = fx["gameweek"].astype(int)
    long = pd.concat([
        pd.DataFrame({"team_id": fx["home_id"], "round": rnd, "gc": fx["away_score"]}),
        pd.DataFrame({"team_id": fx["away_id"], "round": rnd, "gc": fx["home_score"]}),
    ], ignore_index=True)

    rounds = _floats(long["round"])
    gc = _floats(_numeric(long, "gc"))
    gc_asof = np.full(len(long), np.nan)
    for idx in long.groupby("team_id").indices.values():
        W, _ = _asof_weights(rounds[idx], half_life)
        gc_asof[idx] = _weighted_means(W, gc[idx, None])[:, 0]

    return (long.assign(opp_gc_pg_asof=gc_asof)
                .drop_duplicates(["team_id", "round"])
                .sort_values(["team_id", "round"])
                [["team_id", "round", "opp_gc_pg_asof"]]
                .reset_index(drop=True))


def build_training_table(history_all: pd.DataFrame, fixtures: pd.DataFrame,
                         players: pd.DataFrame, profiles: pd.DataFrame,
                         half_life: float = config.RECENCY_HALF_LIFE_GW) -> pd.DataFrame:
    """One row per (player, round) with features + targets.

    history_all : concat of fpl_api.player_history(pid) for all players, each with an
                  added `id` column (player id) and the per-match columns: round, minutes,
                  opponent_team, was_home, total_points, expected_goals/assists/goal_involvements,
                  defensive_contribution, bonus, bps.
    players     : fpl_api.players()  (id, position, team, web_name)
    profiles    : opponent.assign_archetypes(opponent.build_opponent_profiles(...))
    """
    if history_all.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    # stable sort keeps a double gameweek's two matches in their original order
    h = history_all.sort_values(["id", "round"], kind="stable").reset_index(drop=True)
    rounds = _floats(h["round"])
    stats = np.column_stack([_floats(_numeric(h, col)) for col in _FORM_FIELDS.values()])
    mins = _floats(_numeric(h, "minutes"))
    played = (mins > 0).astype(float)
    played_60 = (mins >= 60).astype(float)

    # player form, strictly from earlier rounds (see _asof_weights)
    form = np.full(stats.shape, np.nan)
    games_played = np.zeros(len(h))
    starts_rate = np.full(len(h), np.nan)
    for idx in h.groupby("id").indices.values():
        W, prior = _asof_weights(rounds[idx], half_life)
        form[idx] = _weighted_means(W, stats[idx])
        games_played[idx] = prior @ played[idx]
        starts_rate[idx] = _weighted_means(W, played_60[idx, None])[:, 0]

    meta = players.set_index("id")
    tbl = pd.DataFrame({
        "id": h["id"],
        "round": h["round"].astype(int),
        "web_name": h["id"].map(meta["web_name"]),
        "position": h["id"].map(meta["position"]),
        "was_home": (h["was_home"].fillna(False).astype(bool) if "was_home" in h
                     else pd.Series(False, index=h.index)),
        "opponent_team": _numeric(h, "opponent_team"),
        "target_points": _numeric(h, "total_points"),
        "minutes": _numeric(h, "minutes"),
    })
    tbl[list(_FORM_FIELDS)] = form
    tbl["games_played"] = games_played.astype(int)
    tbl["form_starts_rate"] = starts_rate

    # opponent: leakage-free as-of goals conceded, joined on (opponent, round)
    defence = team_defence_asof(fixtures, half_life).rename(columns={"team_id": "opponent_team"})
    tbl = tbl.merge(defence, on=["opponent_team", "round"], how="left", validate="many_to_one")

    # opponent: season profile. Player at home -> opponent played AWAY -> use the
    # opponent's AWAY defensive strength (and vice versa).
    prof = profiles.set_index("team_id")
    tbl["opp_archetype"] = tbl["opponent_team"].map(prof["archetype"])
    tbl["opp_str_def"] = np.where(tbl["was_home"],
                                  tbl["opponent_team"].map(prof["str_def_away"]),
                                  tbl["opponent_team"].map(prof["str_def_home"]))

    tbl["x_xgi_oppgc"] = tbl["form_xgi_pg"] * tbl["opp_gc_pg_asof"]
    tbl["is_cold"] = tbl["games_played"] < 3

    return _apply_cold_start_priors(tbl[_OUTPUT_COLUMNS])


def _apply_cold_start_priors(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing form (early rounds / debutants) with a position-median prior.

    Leakage-free like every other feature: a row in round t is filled with the
    median form of its position over rows from rounds strictly BEFORE t. Rows with
    nothing earlier to learn from (round 1) stay NaN, and `is_cold` flags them.
    """
    if df.empty:
        return df
    cols = [c for c in _FORM_FIELDS if c in df.columns]
    raw = df[cols].copy()                       # medians come from computed, unfilled form
    for rnd in np.sort(df["round"].unique()):
        rows = (df["round"] == rnd) & raw.isna().any(axis=1)
        if not rows.any():
            continue
        earlier = df["round"] < rnd
        medians = raw[earlier].groupby(df.loc[earlier, "position"]).median()
        prior = medians.reindex(df.loc[rows, "position"]).set_axis(df.index[rows])
        df.loc[rows, cols] = raw.loc[rows].fillna(prior)
    return df
