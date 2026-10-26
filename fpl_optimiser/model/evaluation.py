"""Honest evaluation — every past deadline rebuilt as it stood, every method scored
on the same player-gameweeks.

For each finished gameweek r (from the second one, so there is something to learn
from), everything the planner would have used is rebuilt as of r's deadline:

  the models    refitted on rows strictly before r (earlier seasons, and this
                season's earlier gameweeks), every feature as of r, through the same
                code as the live forecast (predict.forecast_rows)
  the players   those in the game at r, at their club and price at r
  the fixtures  r and the rest of the planning horizon, with later results hidden
  the blend     the planner's: config.py's event-model weight, which is chosen on
                last season (model/tuning.py), so out of sample here
  the pool      the planner's candidate pool as it would have been at r
                (optimise/pool.py on r's own forecast), plus your squad at r if known
  not possible  FPL's injury flags at r (FPL doesn't archive them): replayed
                predictions assume everyone fit; model/forecast_log.py keeps the real
                deadline forecasts from now on

Every method is scored on exactly the same player-gameweeks:

  xp_blend     the planner's blend, at config.EVENT_MODEL_WEIGHT   (the reference)
  blend_wf     the blend with its weight re-chosen at each deadline from earlier
               gameweeks' out-of-sample predictions only (walk-forward); config's
               weight before there are any
  xp           the points model alone        xp_events   the event model alone
  pos_price_p  similar players' (same position and price band) average points per
               appearance x the minutes model's P(plays)
  pos_price    similar players' average points per match (no model at all)
  ewma         recent points: recency-weighted points per match (the half-life)
  season_avg   the player's average points per match so far this season

Uncertainty: the gameweeks, not the player-gameweeks, are the independent units
(hundreds of players share a gameweek's fixtures and conditions), so each method's
per-gameweek scores are compared with the reference's, paired, with a standard
error and a 90% interval across gameweeks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from fpl_optimiser.model import events, inputs, points

METHODS = {
    "xp_blend": "Blend (what the planner uses)",
    "blend_wf": "Blend, weight from earlier weeks",
    "xp": "Points model",
    "xp_events": "Event model",
    "pos_price_p": "Position/price x P(plays)",
    "pos_price": "Position/price",
    "ewma": "Recent points (EWMA)",
    "season_avg": "Season average",
}
REFERENCE = "xp_blend"
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PRICE_BANDS = [0, 4.6, 5.1, 5.6, 6.1, 7.1, 8.6, np.inf]     # £m, lower bound inclusive
POPULATIONS = {"pool": "The planner's pool at each deadline", "all": "Every player"}


@dataclass
class Evaluation:
    season: str
    rounds: list[int]
    first_week: pd.DataFrame            # per (player, fixture) in each deadline's own gameweek
    horizons: dict[int, pd.DataFrame]   # deadline -> its per-fixture forecast over the horizon
    pools: dict[int, set]               # deadline -> the planner's pool as it would have been
    weights: dict[int, float]           # deadline -> blend_wf's event-model weight (walk-forward)
    half_life: float = config.RECENCY_HALF_LIFE_GW
    weight_rounds: dict[int, list[int]] = field(default_factory=dict)   # deadline -> rounds that chose it
    unmatched: int = 0                  # predicted player-fixtures with no match record
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The world as it stood at a deadline
# --------------------------------------------------------------------------- #
def with_fixture_ids(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Match records with FPL's fixture id (FPL always sends it; derived from the
    gameweek, opponent and venue if not)."""
    if "fixture" in history:
        return history
    fx = fixtures.dropna(subset=["gameweek"])
    home = {(int(g), int(a)): int(f) for g, a, f in zip(fx["gameweek"], fx["away_id"], fx["fixture_id"])}
    away = {(int(g), int(h)): int(f) for g, h, f in zip(fx["gameweek"], fx["home_id"], fx["fixture_id"])}
    ids = [(home if wh else away).get((int(r), int(o))) for r, o, wh in
           zip(history["round"], history["opponent_team"], history["was_home"].astype(bool))]
    return history.assign(fixture=ids)


def players_at(cur, r: int) -> pd.DataFrame:
    """The players in the game at gameweek r's deadline, at their club and price
    then (from their match records); anyone first listed later is left out."""
    h = with_fixture_ids(cur.history, cur.fixtures)
    first = h.groupby("id")["round"].min()
    ids = set(first.index[first <= r])
    side = cur.fixtures.set_index("fixture_id")
    hh = h[h["id"].isin(ids) & (h["round"] <= r)].sort_values(["id", "round"], kind="stable")
    team = np.where(hh["was_home"].astype(bool), hh["fixture"].map(side["home_id"]),
                    hh["fixture"].map(side["away_id"]))
    hh = hh.assign(team_at=team,
                   price_at=pd.to_numeric(hh["value"], errors="coerce") / 10 if "value" in hh else np.nan)
    last = hh.groupby("id")[["team_at", "price_at"]].last()
    p = cur.players[cur.players["id"].isin(ids)].copy()
    p["team"] = p["id"].map(last["team_at"]).fillna(p["team"]).astype(int)
    p["price"] = p["id"].map(last["price_at"]).fillna(p["price"]).astype(float)
    names = cur.teams.set_index("id")
    p["team_name"] = p["team"].map(names["name"])
    if "short_name" in names:
        p["team_short"] = p["team"].map(names["short_name"])
    return p.reset_index(drop=True)


def fixtures_at(fixtures: pd.DataFrame, r: int) -> pd.DataFrame:
    """The fixtures as seen at r's deadline: r and later not played yet."""
    f = fixtures.copy()
    later = f["gameweek"] >= r
    f.loc[later, "finished"] = False
    f.loc[later, ["home_score", "away_score"]] = np.nan
    return f


def training_rows(table: pd.DataFrame, season: str, r: int) -> pd.DataFrame:
    """What the models may learn from at r: earlier seasons and this season before r."""
    known = table.dropna(subset=["target_points", "minutes"])
    return known[(known["season"] < season) | ((known["season"] == season) & (known["round"] < r))]


# --------------------------------------------------------------------------- #
# Baselines (as of r, from the training rows only)
# --------------------------------------------------------------------------- #
def _band(price: pd.Series) -> pd.Series:
    return pd.cut(price.astype(float), PRICE_BANDS, right=False)


def position_price(train: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Similar players' average points (same position and price band), weighted like
    the models' training rows: (per match, per appearance), one value per row."""
    t = train.assign(band=_band(train["price"]), w=inputs.weights(train))
    t["wp"] = t["w"] * t["target_points"]

    def means(d: pd.DataFrame):
        g = d.groupby(["position", "band"], observed=True)[["wp", "w"]].sum()
        by_pos = d.groupby("position")[["wp", "w"]].sum()
        return g["wp"] / g["w"], by_pos["wp"] / by_pos["w"]

    keys = list(zip(rows["position"], _band(rows["price"])))
    out = []
    for d in (t, t[t["minutes"] > 0]):
        cell, pos = means(d)
        out.append(pd.Series([cell.get(k, pos.get(k[0], np.nan)) for k in keys], index=rows.index, dtype=float))
    return out[0], out[1]


def season_average(train: pd.DataFrame, season: str, rows: pd.DataFrame) -> pd.Series:
    """Each player's average points per match this season so far (the training rows'
    average for a player with no match yet)."""
    this = train[train["season"] == season]
    avg = this.groupby("id")["target_points"].mean()
    return rows["id"].map(avg).fillna(float(train["target_points"].mean()))


def choose_weight(rows: pd.DataFrame, grid=WEIGHT_GRID) -> float:
    """The event-model weight with the lowest RMSE on these out-of-sample rows."""
    y = rows["target_points"].to_numpy(dtype=float)
    a, b = rows["xp"].to_numpy(dtype=float), rows["xp_events"].to_numpy(dtype=float)
    ok = ~(np.isnan(y) | np.isnan(a) | np.isnan(b))
    rmse = [float(np.sqrt(np.mean(((1 - w) * a[ok] + w * b[ok] - y[ok]) ** 2))) for w in grid]
    return float(grid[int(np.argmin(rmse))])


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #
def run(season: str | None = None, table: pd.DataFrame | None = None,
        rounds: list[int] | None = None, horizon: int = config.PLANNING_HORIZON_GW,
        half_life: float = config.RECENCY_HALF_LIFE_GW,
        event_weight: float = config.EVENT_MODEL_WEIGHT,
        squads: dict[int, list[int]] | None = None, with_pools: bool = True) -> Evaluation:
    """Replay every deadline in `rounds` (default: each finished gameweek after the
    season's first) as it stood. `table` must be multi_season.build_all(half_life=
    half_life). `squads`: deadline -> your squad then (kept in that deadline's pool)."""
    from fpl_optimiser.features import multi_season
    from fpl_optimiser.ingest import season_store
    from fpl_optimiser.model import predict
    from fpl_optimiser.optimise import pool

    season = season or season_store.available()[-1]
    table = multi_season.build_all(half_life=half_life) if table is None else table
    if "half_life" in table and set(table["half_life"].dropna().astype(float)) - {float(half_life)}:
        raise ValueError(f"the training table was built with another half-life than {half_life:g}")
    cur, prior, profiles = multi_season.season_inputs(season)
    prev_name = season_store.previous_season(season)
    prev = season_store.load(prev_name) if prev_name in season_store.available() else None
    done = sorted(int(g) for g in cur.gameweeks.loc[cur.gameweeks["finished"] == True, "id"])  # noqa: E712
    rounds = rounds or done[1:]
    last_gw = int(cur.gameweeks["id"].max())
    actual = (with_fixture_ids(cur.history, cur.fixtures)[["id", "fixture", "total_points", "minutes"]]
              .rename(columns={"fixture": "fixture_id", "total_points": "target_points"}))

    first_weeks, horizons, pools, weights, weight_rounds = [], {}, {}, {}, {}
    unmatched = 0
    for r in rounds:
        train = training_rows(table, season, r)
        model = points.fit(train)
        this_train = train[train["season"] == season]["round"]
        players_r = players_at(cur, r)
        gws = [g for g in range(r, r + horizon) if g <= last_gw]
        fx, ratings, rates = predict.forecast_rows(model, cur, prior, profiles, r, gws, players_r,
                                                   fixtures_at(cur.fixtures, r), prev, half_life)
        fx = fx.join(events.expected_events(fx, ratings, rates))

        # the baselines, from the same training rows
        fx["pos_price"], per_app = position_price(train, fx)
        fx["pos_price_p"] = per_app * fx["p_play"]
        fx["ewma"] = fx["form_points_pg"]
        fx["season_avg"] = season_average(train, season, fx)

        # the planner's blend (config's weight), and the alternative whose weight is
        # chosen from earlier deadlines' out-of-sample first weeks only
        w_r = choose_weight(pd.concat(first_weeks)) if first_weeks else float(event_weight)
        weight_rounds[r] = sorted({int(x) for f in first_weeks for x in f["round"].unique()})
        fx["event_weight"] = float(event_weight)
        fx["half_life"] = float(half_life)
        fx["xp_blend"] = (1 - event_weight) * fx["xp"] + event_weight * fx["xp_events"]
        fx["blend_wf"] = (1 - w_r) * fx["xp"] + w_r * fx["xp_events"]
        fx["trained_to"] = int(this_train.max()) if len(this_train) else 0
        fx["deadline"] = r

        # the planner's pool as it would have been: r's own forecast, pruned
        ids_r: set = set()
        if with_pools:
            per_gw = (fx.groupby(["id", "round"])["xp_blend"].sum().reset_index()
                      .rename(columns={"round": "gw", "xp_blend": "xp"}))
            keep = [i for i in (squads or {}).get(r, []) if i in set(players_r["id"])]
            ids_r = pool.prune(per_gw, players_r[["id", "position", "price"]], keep=keep)

        first = fx[fx["round"] == r].drop(columns=["target_points", "minutes"])
        matched = first.merge(actual, on=["id", "fixture_id"], how="inner")
        unmatched += len(first) - len(matched)
        matched["in_pool"] = matched["id"].isin(ids_r)
        first_weeks.append(matched)
        horizons[r], pools[r], weights[r] = fx, ids_r, w_r

    first_week = pd.concat(first_weeks, ignore_index=True) if first_weeks else pd.DataFrame()
    return Evaluation(season=season, rounds=list(rounds), first_week=first_week, horizons=horizons,
                      pools=pools, weights=weights, half_life=half_life, unmatched=unmatched,
                      weight_rounds=weight_rounds)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def player_gameweeks(ev: Evaluation, population: str = "pool", rounds: list[int] | None = None) -> pd.DataFrame:
    """One row per (player, gameweek): each method's prediction and the actual points
    (a double gameweek's fixtures added up), for the pool or every player."""
    fw = ev.first_week
    if rounds is not None:
        fw = fw[fw["round"].isin(rounds)]
    if population == "pool":
        fw = fw[fw["in_pool"]]
    cols = [c for c in METHODS if c in fw]
    g = fw.groupby(["id", "round"])
    agg = dict(actual=("target_points", "sum"), minutes=("minutes", "sum"), p_play=("p_play", "max"),
               fixtures=("fixture_id", "size"), trained_to=("trained_to", "max"),
               event_weight=("event_weight", "first"), in_pool=("in_pool", "any"))
    agg.update({s: (s, "sum") for s in events.SOURCES if s in fw})
    # a method's prediction for a double gameweek: its fixtures added up, missing if any is
    preds = g[cols].sum().mask(fw[cols].isna().groupby([fw["id"], fw["round"]]).any())
    return pd.concat([preds, g.agg(**agg)], axis=1).reset_index()


def by_gameweek(pgw: pd.DataFrame) -> pd.DataFrame:
    """Each method's scores in each gameweek: n, bias, MAE, RMSE, rank correlation,
    the real points of its top 11, and the hindsight best 11 for reference."""
    rows = []
    for r, d in pgw.groupby("round"):
        best = d.nlargest(11, "actual")["actual"].sum()
        for col, name in METHODS.items():
            if col not in d or d[col].isna().all():
                continue
            e = d[col] - d["actual"]
            with np.errstate(invalid="ignore", divide="ignore"):
                rank = d[col].corr(d["actual"], method="spearman") if d[col].nunique() > 1 else np.nan
            rows.append({"method": col, "label": name, "round": int(r), "n": int(e.notna().sum()),
                         "bias": float(e.mean()), "mae": float(e.abs().mean()),
                         "rmse": float(np.sqrt((e ** 2).mean())), "rank_corr": float(rank),
                         "top11": float(d.nlargest(11, col)["actual"].sum()), "best11": float(best)})
    return pd.DataFrame(rows)


def summary(gw: pd.DataFrame, pgw: pd.DataFrame) -> pd.DataFrame:
    """One row per method: MAE / RMSE / bias over every player-gameweek, and rank
    correlation / top 11 averaged over gameweeks, with their spread across gameweeks."""
    rows = []
    for col, d in gw.groupby("method", sort=False):
        e = pgw[col] - pgw["actual"]
        k = len(d)
        rows.append({"method": col, "label": METHODS[col], "gameweeks": k, "n": int(e.notna().sum()),
                     "bias": float(e.mean()), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                     "rank_corr": float(d["rank_corr"].mean()), "top11": float(d["top11"].mean()),
                     "top11_sd": float(d["top11"].std(ddof=1)) if k > 1 else np.nan,
                     "best11": float(d["best11"].mean()),
                     "captured": float(d["top11"].sum() / d["best11"].sum()) if d["best11"].sum() else np.nan})
    return pd.DataFrame(rows)


def paired(gw: pd.DataFrame, reference: str = REFERENCE, metrics=("rmse", "rank_corr", "top11")) -> pd.DataFrame:
    """The reference against every other method, gameweek by gameweek: the mean
    difference (reference minus method), its standard error and 90% interval across
    gameweeks, and in how many gameweeks the reference was better."""
    from scipy import stats

    ref = gw[gw["method"] == reference].set_index("round")
    rows = []
    for col, d in gw[gw["method"] != reference].groupby("method", sort=False):
        d = d.set_index("round")
        common = ref.index.intersection(d.index)
        for m in metrics:
            diff = (ref.loc[common, m] - d.loc[common, m]).dropna()
            k = len(diff)
            se = float(diff.std(ddof=1) / np.sqrt(k)) if k > 1 else np.nan
            half = float(stats.t.ppf(0.95, k - 1) * se) if k > 1 else np.nan
            better = (diff < 0) if m in ("rmse", "mae") else (diff > 0)
            rows.append({"method": col, "label": METHODS[col], "metric": m, "gameweeks": k,
                         "mean_diff": float(diff.mean()) if k else np.nan, "se": se,
                         "low": float(diff.mean()) - half if k > 1 else np.nan,
                         "high": float(diff.mean()) + half if k > 1 else np.nan,
                         "reference_better": int(better.sum())})
    return pd.DataFrame(rows)


def backtest_table(ev: Evaluation, population: str = "all") -> pd.DataFrame:
    """The summary in the old backtest's shape (method, MAE, RMSE, rank_corr,
    top11_points, gameweeks, predictions), for the Model tab."""
    pgw = player_gameweeks(ev, population)
    s = summary(by_gameweek(pgw), pgw)
    return pd.DataFrame({"method": s["label"], "MAE": s["mae"], "RMSE": s["rmse"], "rank_corr": s["rank_corr"],
                         "top11_points": s["top11"], "gameweeks": s["gameweeks"], "predictions": s["n"]})
