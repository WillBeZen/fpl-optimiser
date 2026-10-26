"""Assemble the model training table: one row per (player, gameweek) played.

Each row carries:
  - targets:   target_points (what Section 3 predicts) and minutes (for the minutes model)
  - player form: recency-weighted aggregates over the player's STRICTLY-PRIOR matches
                 (no leakage — round t only ever sees rounds < t)
  - opponent:  the opponent faced that week — as-of rolling goals conceded (leakage-free),
               plus the season profile's archetype and FPL defensive strength
  - interaction: attacking form x opponent leakiness
  - last season (optional `prior`): the player's previous-season averages blended
               into his form as if played just before GW1, plus prev_* context columns
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

# Last-season context the model also sees (built by features/prior_season.py).
# Rows with no previous-season record get has_prev_season False and NaN elsewhere.
PRIOR_CONTEXT_COLUMNS = [
    "has_prev_season", "prev_points_per_gw", "prev_minutes_per_gw",
    "prev_points_per_90", "prev_xgi_per_90", "prev_defcon_per_90", "prev_changed_team",
]

# Recent minutes, unweighted: who is playing right now. From this season's earlier
# matches only (NaN before the first): minutes in the latest one, the share of the
# last three with 60+ minutes, and how many running he has missed (0 minutes).
RECENT_COLUMNS = ["last_minutes", "starts_last3", "zero_run"]

_OUTPUT_COLUMNS = [
    "id", "round", "web_name", "position", "was_home", "opponent_team", "price",
    "target_points", "minutes",
    *_FORM_FIELDS, "games_played", "form_starts_rate", *RECENT_COLUMNS,
    "opp_gc_pg_asof", "opp_archetype", "opp_str_def", "x_xgi_oppgc", "is_cold",
    *PRIOR_CONTEXT_COLUMNS,
]


def _virtual_last_season(prior: dict, n_gw: int, gap_gw: float) -> pd.DataFrame:
    """Last season as `n_gw` matches at its per-gameweek averages, ending `gap_gw`
    gameweeks before this season's GW1 (its GW38 sits at round -gap_gw)."""
    rows = {"round": np.arange(1, n_gw + 1) - n_gw - gap_gw}
    rows.update({col: prior.get(f"pg_{col}", np.nan) for col in _FORM_FIELDS.values()})
    rows["_played_60"] = prior.get("pg_played_60", np.nan)
    return pd.DataFrame(rows)


def weighted_form_asof(player_history: pd.DataFrame, as_of_round: int,
                       half_life: float = config.RECENCY_HALF_LIFE_GW,
                       prior: dict | None = None,
                       prior_gap_gw: float = config.PREV_SEASON_GAP_GW,
                       prior_n_gw: int = config.GAMEWEEKS_IN_SEASON) -> dict:
    """Recency-weighted form from rounds strictly BEFORE `as_of_round` (no leakage).

    `prior` (optional): this player's previous-season per-gameweek averages
    ({'pg_total_points': ..., 'pg_played_60': ...}, see prior_season.py). Last season
    then counts as `prior_n_gw` extra matches played before GW1, `prior_gap_gw`
    gameweeks of summer break earlier, so it fades as this season's matches arrive.

    This is the one-row reference definition; `build_training_table` computes the
    same values for every row at once (and the tests check they agree).
    """
    h = player_history[player_history["round"] < as_of_round]
    mins = _floats(_numeric(h, "minutes"))
    games_played = int(np.nansum(mins > 0))               # this season's matches only
    h = h.assign(_played_60=(mins >= 60).astype(float))
    if prior is not None:
        h = pd.concat([h, _virtual_last_season(prior, prior_n_gw, prior_gap_gw)],
                      ignore_index=True)
    if h.empty:
        out = {k: float("nan") for k in _FORM_FIELDS}
        out.update(games_played=0, form_starts_rate=float("nan"))
        return out

    g_ago = (as_of_round - h["round"]).to_numpy(dtype=float)
    out: dict = {}
    for feat, col in _FORM_FIELDS.items():
        out[feat] = recency.weighted_mean(_floats(_numeric(h, col)), g_ago, half_life)
    out["games_played"] = games_played
    out["form_starts_rate"] = recency.weighted_mean(_floats(h["_played_60"]), g_ago, half_life)
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


def _weighted_sums(W: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numerator sum(w * v) and denominator sum(w) over the non-NaN values, for
    every row (of W) and column (of values) at once."""
    usable = ~np.isnan(values)
    return W @ np.where(usable, values, 0.0), W @ usable.astype(float)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _weighted_means(W: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`recency.weighted_mean` for every row and column at once (NaN = nothing usable)."""
    return _ratio(*_weighted_sums(W, values))


def _last_season_weight(rounds: np.ndarray, half_life: float, gap_gw: float,
                        n_gw: int) -> np.ndarray:
    """Total recency weight of `_virtual_last_season`'s matches, seen from each round:
    sum over k = 0..n_gw-1 of decay(round + gap_gw + k), in closed form."""
    return (recency.decay_weight(rounds + gap_gw, half_life)
            * recency.decay_weight(np.arange(n_gw), half_life).sum())


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
                         half_life: float = config.RECENCY_HALF_LIFE_GW,
                         prior: pd.DataFrame | None = None,
                         prior_gap_gw: float = config.PREV_SEASON_GAP_GW,
                         prior_n_gw: int = config.GAMEWEEKS_IN_SEASON) -> pd.DataFrame:
    """One row per (player, round) with features + targets.

    history_all : concat of fpl_api.player_history(pid) for all players, each with an
                  added `id` column (player id) and the per-match columns: round, minutes,
                  opponent_team, was_home, total_points, expected_goals/assists/goal_involvements,
                  defensive_contribution, bonus, bps.
    players     : fpl_api.players()  (id, position, team, web_name)
    profiles    : opponent.assign_archetypes(opponent.build_opponent_profiles(...))
    prior       : optional prior_season.prior_table(...) — last season per player
                  (THIS season's ids), blended into form and added as prev_* columns
    """
    if history_all.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    # stable sort keeps a double gameweek's two matches in their original order
    h = history_all.sort_values(["id", "round"], kind="stable").reset_index(drop=True)
    last = prior.set_index("id") if prior is not None else None

    meta = players.set_index("id")
    tbl = pd.DataFrame({
        "id": h["id"],
        "round": h["round"].astype(int),
        "web_name": h["id"].map(meta["web_name"]),
        "position": h["id"].map(meta["position"]),
        "was_home": (h["was_home"].fillna(False).astype(bool) if "was_home" in h
                     else pd.Series(False, index=h.index)),
        "opponent_team": _numeric(h, "opponent_team"),
        "price": _numeric(h, "value") / 10.0,          # that gameweek's price, known pre-match
        "target_points": _numeric(h, "total_points"),
        "minutes": _numeric(h, "minutes"),
    })
    tbl = pd.concat([tbl, _player_form(h, half_life, last, prior_gap_gw, prior_n_gw)], axis=1)

    # opponent: leakage-free as-of goals conceded, joined on (opponent, round)
    defence = team_defence_asof(fixtures, half_life).rename(columns={"team_id": "opponent_team"})
    tbl = tbl.merge(defence, on=["opponent_team", "round"], how="left", validate="many_to_one")

    return _apply_cold_start_priors(_finish(tbl, profiles, last)[_OUTPUT_COLUMNS])


def _recent_minutes(rounds: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """RECENT_COLUMNS for one player's rows, sorted by round: each row sees only the
    matches in strictly earlier rounds (a double gameweek's two rows see the same)."""
    k = np.searchsorted(rounds, rounds, side="left")         # row i's earlier matches: [0, k[i])
    m = np.nan_to_num(mins)
    run = np.zeros(len(m))                                   # matches missed running, up to each one
    for j in range(len(m)):
        run[j] = (run[j - 1] + 1 if j else 1) if m[j] == 0 else 0
    sixty = np.concatenate([[0.0], np.cumsum(m >= 60)])
    out = np.full((len(m), len(RECENT_COLUMNS)), np.nan)
    has = k > 0
    last, n3 = k[has] - 1, np.minimum(k[has], 3)
    out[has, 0] = mins[last]
    out[has, 1] = (sixty[k[has]] - sixty[k[has] - n3]) / n3
    out[has, 2] = run[last]
    return out


def _player_form(h: pd.DataFrame, half_life: float, last: pd.DataFrame | None,
                 prior_gap_gw: float, prior_n_gw: int) -> pd.DataFrame:
    """Form for every row of `h` (any players' histories), strictly from earlier
    rounds (see _asof_weights), with last season blended in where `last` has the
    player (see weighted_form_asof), and recent minutes (_recent_minutes). Indexed
    like `h`."""
    rounds = _floats(h["round"])
    stats = np.column_stack([_floats(_numeric(h, col)) for col in _FORM_FIELDS.values()])
    mins = _floats(_numeric(h, "minutes"))
    played = (mins > 0).astype(float)
    played_60 = (mins >= 60).astype(float)

    last_cols = [f"pg_{col}" for col in _FORM_FIELDS.values()] + ["pg_played_60"]
    form = np.full(stats.shape, np.nan)
    games_played = np.zeros(len(h))
    starts_rate = np.full(len(h), np.nan)
    recent = np.full((len(h), len(RECENT_COLUMNS)), np.nan)
    for pid, idx in h.groupby("id").indices.items():
        W, earlier = _asof_weights(rounds[idx], half_life)
        num, den = _weighted_sums(W, np.column_stack([stats[idx], played_60[idx]]))
        if last is not None and pid in last.index:
            v = last.loc[pid, last_cols].to_numpy(dtype=float)
            w = _last_season_weight(rounds[idx], half_life, prior_gap_gw, prior_n_gw)
            num += np.outer(w, np.nan_to_num(v))
            den += np.outer(w, ~np.isnan(v))
        means = _ratio(num, den)
        form[idx] = means[:, :-1]
        starts_rate[idx] = means[:, -1]
        games_played[idx] = earlier @ played[idx]
        recent[idx] = _recent_minutes(rounds[idx], mins[idx])

    out = pd.DataFrame(form, columns=list(_FORM_FIELDS), index=h.index)
    out["games_played"] = games_played.astype(int)
    out["form_starts_rate"] = starts_rate
    out[RECENT_COLUMNS] = recent
    return out


def _finish(tbl: pd.DataFrame, profiles: pd.DataFrame, last: pd.DataFrame | None) -> pd.DataFrame:
    """The columns training and prediction rows share once form and opponent
    goals-conceded are in place: opponent profile, interaction, cold flag and
    last-season context."""
    # player at home -> opponent played AWAY -> use the opponent's AWAY defensive
    # strength (and vice versa)
    prof = profiles.set_index("team_id")
    tbl["opp_archetype"] = tbl["opponent_team"].map(prof["archetype"])
    tbl["opp_str_def"] = np.where(tbl["was_home"],
                                  tbl["opponent_team"].map(prof["str_def_away"]),
                                  tbl["opponent_team"].map(prof["str_def_home"]))

    tbl["x_xgi_oppgc"] = tbl["form_xgi_pg"] * tbl["opp_gc_pg_asof"]
    tbl["is_cold"] = tbl["games_played"] < 3

    # last-season context (False / NaN for players with no previous-season record)
    ctx = last if last is not None else pd.DataFrame(columns=PRIOR_CONTEXT_COLUMNS)
    for col in PRIOR_CONTEXT_COLUMNS:
        tbl[col] = tbl["id"].map(ctx[col])
    flags = ["has_prev_season", "prev_changed_team"]
    for col in flags:
        tbl[col] = tbl[col].fillna(False).astype(bool)
    rates = [c for c in PRIOR_CONTEXT_COLUMNS if c not in flags]
    tbl[rates] = tbl[rates].astype(float)
    return tbl


def team_defence_at(fixtures: pd.DataFrame, as_of_round: int,
                    half_life: float = config.RECENCY_HALF_LIFE_GW) -> pd.Series:
    """team_id -> recency-weighted goals conceded over finished matches BEFORE
    `as_of_round` (what `team_defence_asof` would give for a match in that round)."""
    fx = fixtures.dropna(subset=["gameweek", "home_score", "away_score"])
    if "finished" in fx.columns:
        fx = fx[fx["finished"] == True]  # noqa: E712
    fx = fx[fx["gameweek"] < as_of_round]
    long = pd.concat([pd.DataFrame({"team_id": fx["home_id"], "gw": fx["gameweek"], "gc": fx["away_score"]}),
                      pd.DataFrame({"team_id": fx["away_id"], "gw": fx["gameweek"], "gc": fx["home_score"]})])
    long["w"] = recency.decay_weight(as_of_round - _floats(long["gw"]), half_life)
    long["wgc"] = long["w"] * _floats(_numeric(long, "gc"))
    sums = long.groupby("team_id")[["wgc", "w"]].sum()
    return sums["wgc"] / sums["w"]


def build_prediction_table(history_all: pd.DataFrame, fixtures: pd.DataFrame,
                           players: pd.DataFrame, profiles: pd.DataFrame,
                           as_of_round: int, gameweeks: list[int],
                           half_life: float = config.RECENCY_HALF_LIFE_GW,
                           prior: pd.DataFrame | None = None,
                           prior_gap_gw: float = config.PREV_SEASON_GAP_GW,
                           prior_n_gw: int = config.GAMEWEEKS_IN_SEASON) -> pd.DataFrame:
    """Feature rows for UPCOMING fixtures: one per (current player, fixture of his
    team in `gameweeks`), with every feature computed exactly as the training table
    would for a match in round `as_of_round`, from rounds strictly before it.

    A double gameweek gives a player two rows; a blank gameweek none. Targets are
    unknown, so `target_points` and `minutes` are NaN. `round` is the fixture's
    gameweek; `fixture_id` and `team` identify it.
    """
    last = prior.set_index("id") if prior is not None else None

    # form as of `as_of_round`: one probe row per current player, in that round
    past = history_all[history_all["round"] < as_of_round] if len(history_all) else history_all
    probes = pd.DataFrame({"id": players["id"].to_numpy(), "round": as_of_round, "_probe": True})
    h = (pd.concat([past, probes], ignore_index=True)
           .sort_values(["id", "round"], kind="stable").reset_index(drop=True))
    is_probe = h["_probe"].eq(True).to_numpy()
    all_form = _player_form(h, half_life, last, prior_gap_gw, prior_n_gw)
    form = all_form[is_probe].set_axis(h.loc[is_probe, "id"].to_numpy())

    # each current player against each upcoming fixture of his team
    fx = fixtures[fixtures["gameweek"].isin(gameweeks)]
    if "finished" in fx.columns:
        fx = fx[fx["finished"] != True]  # noqa: E712
    sides = pd.concat([
        pd.DataFrame({"team": fx["home_id"], "opponent_team": fx["away_id"], "was_home": True,
                      "round": fx["gameweek"], "fixture_id": fx["fixture_id"]}),
        pd.DataFrame({"team": fx["away_id"], "opponent_team": fx["home_id"], "was_home": False,
                      "round": fx["gameweek"], "fixture_id": fx["fixture_id"]}),
    ], ignore_index=True)
    cols = ["id", "web_name", "position", "team", "price"]
    tbl = players[cols].merge(sides, on="team").sort_values(["id", "round", "fixture_id"])
    tbl = tbl.reset_index(drop=True)
    tbl["round"] = tbl["round"].astype(int)
    tbl["target_points"] = np.nan
    tbl["minutes"] = np.nan
    tbl = tbl.join(form, on="id")
    tbl["opp_gc_pg_asof"] = tbl["opponent_team"].map(team_defence_at(fixtures, as_of_round, half_life))
    tbl = _finish(tbl, profiles, last)

    # cold start, exactly as _apply_cold_start_priors would for a match in this
    # round: the position's median form over every earlier player-match row
    earlier = all_form[~is_probe].assign(
        position=h.loc[~is_probe, "id"].map(players.set_index("id")["position"]).to_numpy())
    medians = earlier.groupby("position")[list(_FORM_FIELDS)].median()
    for col in _FORM_FIELDS:
        tbl[col] = tbl[col].fillna(tbl["position"].map(medians[col]) if len(medians) else np.nan)
    return tbl[_OUTPUT_COLUMNS + ["team", "fixture_id"]]


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
