"""Explainers — what each stage of the prediction engine is doing, as tables.

The report's "The model" tab walks through the pipeline stage by stage; these
functions produce the numbers it shows. None of them changes a prediction.

  weight_curve        how much a match counts, by how long ago it was (half-life)
  prior_share         how much of a regular's form is still last season, by gameweek
  form_story          one player's matches, each with its recency weight, and the
                      form they add up to (vs a plain average)
  half_life_table     what players' form would be with a shorter or longer half-life
  feature_importance  which inputs each model leans on (permutation importance)
  opponent_response   how the points model's points-if-he-plays move with the
                      opponent's defence, by position
  opponent_matrix     each player's expected points against every opponent
  matchup_history     what each player type has actually scored against each
                      opponent style this season
  flag_impact         who FPL's injury and suspension flags cut, and by how much
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from fpl_optimiser.features import build, recency
from fpl_optimiser.model import events, inputs, minutes

FEATURE_LABELS = {
    "price": "Price", "form_points_pg": "Form: points", "form_minutes_pg": "Form: minutes",
    "form_xg_pg": "Form: xG", "form_xa_pg": "Form: xA", "form_xgi_pg": "Form: xGI",
    "form_defcon_pg": "Form: def. contributions", "form_bonus_pg": "Form: bonus",
    "form_bps_pg": "Form: BPS", "form_starts_rate": "Form: 60+ min rate",
    "games_played": "Games played", "opp_gc_pg_asof": "Opponent goals conceded",
    "opp_archetype": "Opponent style", "opp_str_def": "Opponent defence rating (FPL)",
    "x_xgi_oppgc": "Form xGI x opponent leakiness", "prev_points_per_gw": "Last season: points",
    "prev_minutes_per_gw": "Last season: minutes", "prev_points_per_90": "Last season: points/90",
    "prev_xgi_per_90": "Last season: xGI/90", "prev_defcon_per_90": "Last season: def. contr./90",
    "was_home": "Home", "is_cold": "Few games yet", "has_prev_season": "Has last season",
    "prev_changed_team": "Changed club", "earlier_season": "Row from an earlier season",
    "position": "Position",
}


# --------------------------------------------------------------------------- #
# Recency weighting
# --------------------------------------------------------------------------- #
def weight_curve(as_of_round: int, half_life: float = config.RECENCY_HALF_LIFE_GW,
                 gap_gw: float = config.PREV_SEASON_GAP_GW) -> pd.DataFrame:
    """The weight each of this season's rounds gets when predicting `as_of_round`, and
    last season's block, with each one's share of the total (a player who played
    every match)."""
    rounds = np.arange(1, as_of_round)
    rows = pd.DataFrame({"label": [f"GW{r}" for r in rounds], "round": rounds,
                         "gameweeks_ago": as_of_round - rounds,
                         "weight": recency.decay_weight(as_of_round - rounds, half_life)})
    last = float(events.prior_weight(as_of_round, half_life, gap_gw))
    rows = pd.concat([pd.DataFrame({"label": ["Last season"], "round": [0],
                                    "gameweeks_ago": [np.nan], "weight": [last]}), rows],
                     ignore_index=True)
    rows["share"] = rows["weight"] / rows["weight"].sum()
    return rows


def prior_share(half_life: float = config.RECENCY_HALF_LIFE_GW,
                gap_gw: float = config.PREV_SEASON_GAP_GW,
                rounds=range(1, config.GAMEWEEKS_IN_SEASON + 1)) -> pd.DataFrame:
    """Last season's share of a regular's form, gameweek by gameweek."""
    out = []
    for r in rounds:
        this = recency.decay_weight(r - np.arange(1, r), half_life).sum()
        last = float(events.prior_weight(r, half_life, gap_gw))
        out.append({"gw": r, "share": last / (last + this)})
    return pd.DataFrame(out)


def form_story(history: pd.DataFrame, player_id: int, as_of_round: int,
               prior: pd.DataFrame | None = None, teams: pd.DataFrame | None = None,
               half_life: float = config.RECENCY_HALF_LIFE_GW) -> tuple[pd.DataFrame, dict]:
    """One player's matches before `as_of_round`, each with its recency weight and
    share of the form, plus a summary: recency-weighted form (with last season, as
    the model sees it), this season's plain average, and last season's average."""
    h = history[(history["id"] == player_id) & (history["round"] < as_of_round)].sort_values("round")
    w = recency.decay_weight(as_of_round - h["round"].to_numpy(dtype=float), half_life)
    short = teams.set_index("id")["short_name"] if teams is not None else pd.Series(dtype=str)
    matches = pd.DataFrame({
        "round": h["round"].to_numpy(),
        "opponent": h["opponent_team"].map(short).fillna("").to_numpy()
                    + np.where(h["was_home"].astype(bool), " (H)", " (A)"),
        "minutes": pd.to_numeric(h["minutes"], errors="coerce").to_numpy(),
        "points": pd.to_numeric(h["total_points"], errors="coerce").to_numpy(),
        "xgi": pd.to_numeric(h.get("expected_goal_involvements"), errors="coerce").to_numpy()
               if "expected_goal_involvements" in h else np.nan,
        "weight": w})
    p = None
    if prior is not None and len(prior) and player_id in set(prior["id"]):
        p = prior.set_index("id").loc[player_id].to_dict()
    with_prior = build.weighted_form_asof(h, as_of_round, half_life, prior=p)
    last_w = float(events.prior_weight(as_of_round, half_life)) if p else 0.0
    total = matches["weight"].sum() + last_w
    matches["share"] = matches["weight"] / total if total else np.nan
    summary = {"form": with_prior["form_points_pg"],
               "plain_average": float(matches["points"].mean()) if len(matches) else float("nan"),
               "last_season": p["pg_total_points"] if p else float("nan"),
               "last_season_share": last_w / total if total else float("nan"),
               "matches": len(matches)}
    return matches, summary


def half_life_table(history: pd.DataFrame, ids, as_of_round: int, prior: pd.DataFrame | None = None,
                    half_lives=(4.0, config.RECENCY_HALF_LIFE_GW, 16.0)) -> pd.DataFrame:
    """Form (points per gameweek, as the model sees it) under different half-lives."""
    pri = prior.set_index("id") if prior is not None and len(prior) else None
    rows = []
    for pid in ids:
        h = history[history["id"] == pid]
        p = pri.loc[pid].to_dict() if pri is not None and pid in pri.index else None
        row = {"id": pid}
        for hl in half_lives:
            row[f"half-life {hl:g}"] = build.weighted_form_asof(h, as_of_round, hl, prior=p)["form_points_pg"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("id")


# --------------------------------------------------------------------------- #
# What the models lean on
# --------------------------------------------------------------------------- #
def feature_importance(model, rows: pd.DataFrame, n_repeats: int = 3, max_rows: int = 3000,
                       seed: int = 0) -> pd.DataFrame:
    """Permutation importance: how much worse each model gets when one input is
    shuffled (so it carries no information). Points model: rise in RMSE of points-
    if-he-plays, on matches played; minutes model: rise in the error of P(plays).

    rows: training-table rows with targets (e.g. this season's)."""
    rng = np.random.default_rng(seed)
    rows = rows.dropna(subset=["target_points", "minutes"])
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=seed)
    X = inputs.design_matrix(rows)
    played = (rows["minutes"] > 0).to_numpy()
    y_pts = rows["target_points"].to_numpy(dtype=float)
    y_play = played.astype(float)

    def scores(Xs: pd.DataFrame) -> tuple[float, float]:
        pts = model.if_played.predict(Xs[played]) if played.any() else np.array([])
        rmse = float(np.sqrt(np.mean((pts - y_pts[played]) ** 2))) if played.any() else np.nan
        p = minutes._positive_proba(model.minutes.play, Xs)
        brier = float(np.mean((p - y_play) ** 2))
        return rmse, brier

    base_rmse, base_brier = scores(X)
    out = []
    for col in inputs.FEATURES:
        d_rmse, d_brier = [], []
        for _ in range(n_repeats):
            Xs = X.copy()
            Xs[col] = Xs[col].to_numpy()[rng.permutation(len(Xs))]
            if col == "position":
                Xs[col] = pd.Categorical(Xs[col], categories=inputs.POSITIONS)
            r, b = scores(Xs)
            d_rmse.append(r - base_rmse)
            d_brier.append(b - base_brier)
        out.append({"feature": col, "label": FEATURE_LABELS.get(col, col),
                    "points": float(np.mean(d_rmse)), "minutes": float(np.mean(d_brier))})
    return pd.DataFrame(out).sort_values("points", ascending=False).reset_index(drop=True)


def _with_opponent(rows: pd.DataFrame, opponent: int, home: bool, profiles: pd.DataFrame,
                   defence: pd.Series) -> pd.DataFrame:
    """Prediction rows re-pointed at another opponent (and venue), with every
    opponent feature recomputed the way build._finish does."""
    prof = profiles.set_index("team_id")
    r = rows.copy()
    r["opponent_team"] = opponent
    r["was_home"] = home
    r["opp_gc_pg_asof"] = defence.get(opponent, np.nan)
    r["opp_archetype"] = prof["archetype"].get(opponent, np.nan)
    r["opp_str_def"] = prof["str_def_away" if home else "str_def_home"].get(opponent, np.nan)
    r["x_xgi_oppgc"] = r["form_xgi_pg"] * r["opp_gc_pg_asof"]
    return r


def opponent_response(model, rows: pd.DataFrame, grid=None) -> pd.DataFrame:
    """The points model's average points-if-he-plays, by position, as the opponent's
    goals conceded per game (the as-of feature) moves across `grid`, everything else
    about each player held as it is."""
    grid = np.round(np.arange(0.6, 2.61, 0.2), 2) if grid is None else grid
    out = []
    for g in grid:
        r = rows.assign(opp_gc_pg_asof=g, x_xgi_oppgc=rows["form_xgi_pg"] * g)
        pts = model.if_played.predict(inputs.design_matrix(r))
        for pos, v in pd.Series(pts, index=r.index).groupby(r["position"]).mean().items():
            out.append({"position": pos, "opp_gc_pg": g, "pts_if_play": v})
    return pd.DataFrame(out)


def opponent_matrix(fc, ids, profiles: pd.DataFrame, fixtures: pd.DataFrame,
                    teams: pd.DataFrame) -> pd.DataFrame:
    """Each player's expected points (the blend the planner uses, assuming he's fit)
    against every other club, averaged over home and away: a scout's "who does he
    score against" grid. Rows are the player's next fixture's features, re-pointed."""
    fx = fc.per_fixture
    base = fx[fx["id"].isin(ids)].sort_values(["round", "fixture_id"]).drop_duplicates("id")
    defence = build.team_defence_at(fixtures, fc.next_gw)
    w = fc.event_weight
    out = []
    for opp in teams["id"]:
        both = []
        for home in (True, False):
            r = _with_opponent(base, int(opp), home, profiles, defence)
            r = r[r["team"] != opp]
            if r.empty:
                continue
            pred = fc.model.predict(r)
            ev = events.expected_events(r.assign(p_play=pred["p_play"], p_60=pred["p_60"]),
                                        fc.ratings, fc.rates)
            both.append(pd.Series((1 - w) * pred["xp"].to_numpy() + w * ev["xp_events"].to_numpy(),
                                  index=r["id"].to_numpy()))
        if both:
            avg = pd.concat(both, axis=1).mean(axis=1)
            out += [{"id": pid, "opponent": int(opp), "xp": v} for pid, v in avg.items()]
    if not out:
        return pd.DataFrame()
    m = pd.DataFrame(out).pivot(index="id", columns="opponent", values="xp")
    m.columns = m.columns.map(teams.set_index("id")["short_name"])
    return m


def matchup_history(table: pd.DataFrame, player_types: pd.Series, profiles: pd.DataFrame) -> pd.DataFrame:
    """Points per appearance this season by (position, player type, opponent style):
    what each kind of player has actually scored against each kind of defence.

    table        training rows (one season), with opp_archetype, position, minutes
    player_types id -> type name (player_profiles.assign_archetypes)
    profiles     opponent profiles with archetype (0 = meanest defence)"""
    rows = table[(table["minutes"] > 0)].copy()
    rows["type"] = rows["id"].map(player_types)
    rows = rows.dropna(subset=["type", "opp_archetype"])
    names = style_names(profiles)
    rows["style"] = rows["opp_archetype"].astype(int).map(names)
    return (rows.groupby(["position", "type", "style"])
            .agg(points=("target_points", "mean"), matches=("target_points", "size"))
            .reset_index())


def style_names(profiles: pd.DataFrame) -> dict[int, str]:
    """Opponent archetype -> a plain name, ordered from tightest to leakiest defence."""
    k = int(profiles["archetype"].max()) + 1
    words = {2: ["Tight defences", "Leaky defences"],
             3: ["Tight defences", "Middling defences", "Leaky defences"]}.get(
        k, [f"Defence style {i + 1}" for i in range(k)])
    return dict(enumerate(words))


def flag_impact(fc, n: int = 20) -> pd.DataFrame:
    """Next gameweek: the players FPL's flags cut most (expected points before vs after)."""
    nxt = fc.xp[fc.xp["gw"] == fc.next_gw].assign(cut=lambda d: d["xp_raw"] - d["xp"])
    return nxt[nxt["cut"] > 0.05].nlargest(n, "cut")[
        ["id", "web_name", "team_name", "position", "status", "news", "xp_raw", "xp", "cut"]]
