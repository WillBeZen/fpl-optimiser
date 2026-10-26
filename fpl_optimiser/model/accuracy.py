"""Model accuracy — how predicted points lined up with what players actually scored.

Over the last config.ACCURACY_WINDOW_GW finished gameweeks, each player's predicted
points come from the walk-forward replay (model/backtest.py): for every gameweek the
models are refitted on data strictly before it, every recency weight uses the
half-life in config.py, and the prediction is the same blend the planner uses. So
each number is what the model would have said at that deadline. (FPL's injury flags
at past deadlines aren't archived, so the replay can't apply them; the forecasts
logged at each deadline from now on can: model/forecast_log.py.)

Actual points are FPL's own, from the filed match records, which `reconcile` checks
against FPL's official gameweek totals (the live endpoint) and season totals.

The players are the planner's candidate pool (optimise/pool.py: everyone not
dominated, plus your squad).

  window            the gameweeks to check
  player_gameweeks  one row per (player, gameweek): predicted, actual, minutes, sources
  summary           totals, bias, typical miss, rank correlation, top-11 captured, ...
  by_gameweek, by_position, calibration, by_source, misses, minutes_split
  half_life_check   the same window replayed at other half-lives
  reconcile         checks that the numbers add up and match FPL
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config
from fpl_optimiser.model import events

# FPL's scoring identifiers (event/<gw>/live) -> the event model's sources
LIVE_SOURCES = {"pts_minutes": "Appearance", "pts_goals_scored": "Goals", "pts_assists": "Assists",
                "pts_clean_sheets": "Clean sheet", "pts_goals_conceded": "Conceded", "pts_saves": "Saves",
                "pts_defensive_contribution": "Def. contributions", "pts_bonus": "Bonus",
                "pts_yellow_cards": "Cards", "pts_red_cards": "Cards"}
CALIBRATION_EDGES = (0, 1, 2, 3, 4, 5, 6, 8, np.inf)


def window(gameweeks: pd.DataFrame, n: int = config.ACCURACY_WINDOW_GW) -> list[int]:
    """The last `n` gameweeks FPL has marked finished, oldest first."""
    done = gameweeks.loc[gameweeks["finished"] == True, "id"].astype(int).sort_values()  # noqa: E712
    return [int(g) for g in done.iloc[-n:]]


def predicted_column(preds: pd.DataFrame) -> str:
    """The blend the planner uses when the replay has it, else the points model."""
    return "xp_blend" if "xp_blend" in preds and preds["xp_blend"].notna().any() else "xp"


def player_gameweeks(preds: pd.DataFrame, rounds: list[int], ids=None) -> pd.DataFrame:
    """One row per (player, gameweek) in `rounds`: predicted and actual points (a
    double gameweek's fixtures added up), minutes, P(plays), the two models' parts
    and the event model's points by source. `ids` limits it to those players."""
    p = preds[preds["round"].isin(rounds)]
    if ids is not None:
        p = p[p["id"].isin(set(ids))]
    col = predicted_column(p)
    agg = {"predicted": (col, lambda s: s.sum(min_count=len(s))),     # any missing part -> NaN
           "actual": ("target_points", "sum"), "minutes": ("minutes", "sum"),
           "p_play": ("p_play", "max"), "fixtures": ("round", "size"),
           "points_model": ("xp", "sum"), "trained_to": ("trained_to", "max")}
    if "xp_events" in p:
        agg["event_model"] = ("xp_events", "sum")
        agg.update({s: (s, "sum") for s in events.SOURCES if s in p})
    out = p.groupby(["id", "round"]).agg(**agg).reset_index()
    out["error"] = out["predicted"] - out["actual"]
    return out


def _weights(rounds: pd.Series, half_life: float) -> pd.Series:
    """The half-life's weight for each gameweek, the latest counting 1."""
    return 0.5 ** ((rounds.max() - rounds) / half_life)


def summary(pg: pd.DataFrame, half_life: float = config.RECENCY_HALF_LIFE_GW) -> dict:
    """Headline numbers for a set of player-gameweeks.

    bias      mean of (predicted - actual): + = the model expected too much
    mae/rmse  typical miss per player-gameweek (RMSE punishes big misses more)
    rank_corr how well it ordered players within each gameweek (Spearman, averaged)
    top11     real points of the 11 it rated highest, per gameweek; best11 = the
              11 who actually scored most (hindsight); captured = top11 / best11
    weighted  MAE and bias with each gameweek weighted by the half-life"""
    if pg.empty:
        return {"player_gameweeks": 0}
    err = pg["error"]
    w = _weights(pg["round"], half_life)
    by_gw = pg.groupby("round")
    with warnings.catch_warnings():               # one point or a constant column: no correlation (NaN)
        warnings.simplefilter("ignore", RuntimeWarning)
        rank = by_gw[["predicted", "actual"]].apply(lambda d: d["predicted"].corr(d["actual"], method="spearman"))
        corr = float(pg["predicted"].corr(pg["actual"]))
    top = by_gw.apply(lambda d: d.nlargest(11, "predicted")["actual"].sum(), include_groups=False)
    best = by_gw.apply(lambda d: d.nlargest(11, "actual")["actual"].sum(), include_groups=False)
    return {"player_gameweeks": len(pg), "players": pg["id"].nunique(), "gameweeks": pg["round"].nunique(),
            "predicted": float(pg["predicted"].sum()), "actual": float(pg["actual"].sum()),
            "bias": float(err.mean()), "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err ** 2).mean())), "corr": corr,
            "rank_corr": float(rank.mean()), "top11": float(top.mean()), "best11": float(best.mean()),
            "captured": float(top.sum() / best.sum()) if best.sum() else float("nan"),
            "within_2": float((err.abs() <= 2).mean()),
            "weighted_mae": float((w * err.abs()).sum() / w.sum()),
            "weighted_bias": float((w * err).sum() / w.sum())}


def by_gameweek(pg: pd.DataFrame, half_life: float = config.RECENCY_HALF_LIFE_GW) -> pd.DataFrame:
    """One row per gameweek: its half-life weight and the summary's numbers."""
    w = _weights(pd.Series(sorted(pg["round"].unique())), half_life)
    rows = []
    for (gw, d), weight in zip(pg.groupby("round"), w):
        s = summary(d, half_life)
        rows.append({"GW": int(gw), "weight": float(weight), "players": s["players"], "predicted": s["predicted"],
                     "actual": s["actual"], "difference": s["predicted"] - s["actual"], "bias": s["bias"],
                     "mae": s["mae"], "rmse": s["rmse"], "rank_corr": s["rank_corr"], "top11": s["top11"],
                     "best11": s["best11"], "captured": s["captured"]})
    return pd.DataFrame(rows)


def by_position(pg: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Accuracy per position (positions: id -> GK/DEF/MID/FWD)."""
    d = pg.assign(position=pg["id"].map(positions))
    rows = []
    for pos in [p for p in config.POSITION_QUOTA if p in set(d["position"])]:
        g = d[d["position"] == pos]
        rows.append({"position": pos, "player_gameweeks": len(g), "predicted": g["predicted"].mean(),
                     "actual": g["actual"].mean(), "bias": g["error"].mean(), "mae": g["error"].abs().mean(),
                     "rmse": float(np.sqrt((g["error"] ** 2).mean()))})
    return pd.DataFrame(rows)


def calibration(pg: pd.DataFrame, edges=CALIBRATION_EDGES) -> pd.DataFrame:
    """Player-gameweeks grouped by what was predicted: when the model said 3-4 points,
    what did those players actually average? (A calibrated model's two columns match.)"""
    labels = [f"{a:g}-{b:g}" if np.isfinite(b) else f"{a:g}+" for a, b in zip(edges[:-1], edges[1:])]
    d = pg.assign(bin=pd.cut(pg["predicted"], list(edges), right=False, labels=labels))
    out = (d.groupby("bin", observed=True)
           .agg(n=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean"),
                returned=("actual", lambda a: (a >= 4).mean()))
           .reset_index())
    return out


def by_source(pg: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """The event model's predicted points by scoring source against FPL's actual
    breakdown (event/<gw>/live), over the same player-gameweeks."""
    if live is None or live.empty or "Goals" not in pg:
        return pd.DataFrame()
    keys = pg[["id", "round"]].rename(columns={"round": "gw"})
    act = keys.merge(live, on=["id", "gw"], how="left")
    rows = []
    for src in events.SOURCES:
        cols = [c for c, s in LIVE_SOURCES.items() if s == src and c in act]
        rows.append({"source": src, "predicted": float(pg[src].sum()),
                     "actual": float(act[cols].fillna(0).to_numpy().sum()) if cols else 0.0})
    other = [c for c in act.columns if c.startswith("pts_") and c not in LIVE_SOURCES]
    if other:
        rows.append({"source": "Other (own goals, penalties)", "predicted": 0.0,
                     "actual": float(act[other].fillna(0).to_numpy().sum())})
    out = pd.DataFrame(rows)
    out["difference"] = out["predicted"] - out["actual"]
    return out


def minutes_split(pg: pd.DataFrame) -> dict:
    """How much of the total miss came from players who didn't play at all."""
    total = pg["error"].abs().sum()
    absent = pg.loc[pg["minutes"] == 0, "error"].abs().sum()
    return {"didnt_play": int((pg["minutes"] == 0).sum()), "played": int((pg["minutes"] > 0).sum()),
            "share_from_absent": float(absent / total) if total else 0.0}


def misses(pg: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The biggest over-predictions (scored far less) and under-predictions (far more),
    each with a plain reason."""
    def reason(r) -> str:
        if r["error"] > 0:
            if r["minutes"] == 0:
                return "Didn't play"
            return "Played under 60 minutes" if r["minutes"] < 60 * r["fixtures"] else "Played, few returns"
        return "Haul" if r["actual"] >= 10 else "Returned more than expected"
    over = pg.nlargest(n, "error").copy()
    under = pg.nsmallest(n, "error").copy()
    for d in (over, under):
        d["reason"] = [reason(r) for _, r in d.iterrows()]
    return over, under


def reference_rows(ev, rounds: list[int] | None = None, population: str = "pool") -> pd.DataFrame:
    """The harness's (model/evaluation.py) player-gameweeks in this module's shape:
    `predicted` is the planner's blend (the reference), plus the actual points, its
    error, both models' parts and sources."""
    from fpl_optimiser.model import evaluation

    pg = evaluation.player_gameweeks(ev, population, rounds)
    pg = pg.rename(columns={"xp_blend": "predicted", "xp": "points_model", "xp_events": "event_model"})
    pg["error"] = pg["predicted"] - pg["actual"]
    return pg


def half_life_check(ev, half_lives, rounds: list[int] | None = None) -> pd.DataFrame:
    """The same deadlines replayed at each half-life (every form feature and the event
    model's rates rebuilt with it), scored on the same player-gameweeks as `ev`
    (the pools it built at each deadline). Only the deadline's own gameweek is
    forecast, so it is quicker than the full replay."""
    from fpl_optimiser.features import multi_season
    from fpl_optimiser.model import evaluation

    rounds = rounds or ev.rounds
    keys = ev.first_week[ev.first_week["in_pool"] & ev.first_week["round"].isin(rounds)][["id", "round"]]
    keys = keys.drop_duplicates()
    rows = []
    for h in half_lives:
        if h == ev.half_life:
            run = ev
        else:
            run = evaluation.run(ev.season, multi_season.build_all(half_life=h), rounds=rounds, horizon=1,
                                 half_life=h, with_pools=False)
        fw = run.first_week.merge(keys, on=["id", "round"])
        fw = fw.assign(in_pool=True)
        sub = evaluation.Evaluation(season=ev.season, rounds=rounds, first_week=fw, horizons={}, pools={},
                                    weights=run.weights, half_life=h)
        pg = reference_rows(sub, rounds, "pool")
        rows.append({"half_life": h, "current": h == config.RECENCY_HALF_LIFE_GW, **summary(pg, h)})
    return pd.DataFrame(rows)


def reconcile(ev, season, rounds: list[int] | None = None, pg: pd.DataFrame | None = None,
              gw_table: pd.DataFrame | None = None) -> list[dict]:
    """Data checks for the replay (not a score of the model): each {check, ok
    (True / False / None = can't tell), detail}. `ev` is the harness's Evaluation,
    `season` the filed Season (FPL's official gameweek totals, pending matches and
    season totals)."""
    checks = []
    rounds = rounds or ev.rounds
    fw = ev.first_week[ev.first_week["round"].isin(rounds)]
    pg = reference_rows(ev, rounds, "pool") if pg is None else pg

    def add(check, ok, detail):
        checks.append({"check": check, "ok": None if ok is None else bool(ok), "detail": detail})

    gws = season.gameweeks.set_index("id")
    finished = [g for g in rounds if bool(gws.at[g, "finished"])]
    checked = [g for g in rounds if bool(gws.get("data_checked", gws["finished"]).get(g, False))]
    add("Every replayed gameweek is finished", len(finished) == len(rounds) == len(checked),
        f"GW{rounds[0]}-{rounds[-1]}: {len(finished)} of {len(rounds)} finished, {len(checked)} checked by FPL"
        if rounds else "no finished gameweeks yet")

    live = season.live
    have = sorted(set(live["gw"]) & set(rounds)) if len(live) else []
    if not have:
        add("Actual points match FPL's official gameweek totals", None,
            "FPL's gameweek totals aren't filed yet (they're filed by a data pull)")
    else:
        ours = fw[fw["round"].isin(have)].groupby(["id", "round"])[["target_points", "minutes"]].sum()
        theirs = (live[live["gw"].isin(have)].rename(columns={"gw": "round"})
                  .set_index(["id", "round"])[["total_points", "minutes"]].reindex(ours.index))
        same = (ours["target_points"] == theirs["total_points"]) & (ours["minutes"] == theirs["minutes"])
        add("Actual points match FPL's official gameweek totals", bool(same.all()),
            f"{int(same.sum())} of {len(same)} replayed player-gameweeks identical (points and minutes) "
            f"in GW{have[0]}-{have[-1]}")
        allgw = sorted(set(live["gw"]))
        h = season.history[season.history["round"].isin(allgw)].groupby("id")["total_points"].sum()
        lv = live[live["gw"].isin(allgw)].groupby("id")["total_points"].sum()
        both = pd.concat([h.rename("ours"), lv.rename("fpl")], axis=1).fillna(0)
        n_same = int((both["ours"] == both["fpl"]).sum())
        add("Every finished gameweek this season matches FPL", n_same == len(both),
            f"{n_same} of {len(both)} players' points over GW{allgw[0]}-{allgw[-1]} equal FPL's official totals")
        add("FPL's own breakdown adds up", bool(live["explained"].all()),
            f"each player's points by source sum to his total in {int(live['explained'].sum())} of {len(live)} cases")

    tot = pd.concat([season.history, season.pending])
    mine = tot.groupby("id")["total_points"].sum()
    fpl_tot = season.players.set_index("id")["total_points"].reindex(mine.index)
    behind = int((mine != fpl_tot).sum())
    add("Match records are up to date", behind == 0,
        f"all {len(mine)} players' match records add up to FPL's season totals" if behind == 0 else
        f"{behind} of {len(mine)} players have points FPL already counts that the filed match records don't "
        "have yet (a refresh fetches them); finished gameweeks are unaffected")

    late = fw[fw["trained_to"] >= fw["round"]]
    add("No prediction used later data", late.empty,
        "each deadline's models, features, baselines and pool came only from earlier gameweeks"
        if late.empty else f"{len(late)} predictions used their own gameweek or later")

    wrong = [r for r, src in ev.weight_rounds.items() if any(s >= r for s in src)] if ev.weight_rounds else []
    add("Blend weights chosen from earlier weeks only", not wrong,
        "the 'weight from earlier weeks' blend, at each deadline: " + ", ".join(
            f"GW{r} {w:g}" for r, w in ev.weights.items() if r in rounds)
        if not wrong else f"deadlines {wrong} used their own or later weeks")

    gap = (fw["xp_blend"] - ((1 - fw["event_weight"]) * fw["xp"] + fw["event_weight"] * fw["xp_events"])).abs().max()
    missing = int(fw["xp_blend"].isna().sum())
    weights = sorted(set(fw["event_weight"].dropna().astype(float)))
    add("Predictions are the planner's blend", missing == 0 and (np.isnan(gap) or gap < 1e-9),
        "(1 - w) x points model + w x event model for every prediction, w = "
        + ", ".join(f"{w:g}" for w in weights) + f" (config.py: {config.EVENT_MODEL_WEIGHT:g})"
        + (f"; {missing} missing" if missing else ""))

    used = set(ev.first_week["half_life"].dropna().astype(float)) if "half_life" in ev.first_week else set()
    used = used or {float(ev.half_life)}
    add("Predictions use the half-life in config.py", used == {float(config.RECENCY_HALF_LIFE_GW)},
        "form features and event-model rates built with a half-life of "
        + ", ".join(f"{h:g}" for h in sorted(used)) + f" gameweeks (config.py: {config.RECENCY_HALF_LIFE_GW:g})")

    pool_rounds = [r for r in rounds if r in ev.pools]
    add("Pools rebuilt at each deadline", len(pool_rounds) == len(rounds) and all(ev.pools[r] for r in pool_rounds),
        "each deadline's pool pruned from that deadline's own forecast: "
        + ", ".join(f"GW{r} {len(ev.pools[r])}" for r in pool_rounds))

    add("Every prediction has a match record", ev.unmatched == 0,
        "every predicted player-fixture was played (or sat out) and is in FPL's records"
        if ev.unmatched == 0 else f"{ev.unmatched} predicted player-fixtures have no match record")

    sums = [float(pg["predicted"].sum()), float(fw[fw["in_pool"]]["xp_blend"].sum())]
    acts = [float(pg["actual"].sum()), float(fw[fw["in_pool"]]["target_points"].sum())]
    if gw_table is not None and len(gw_table):
        sums.append(float(gw_table["predicted"].sum()))
        acts.append(float(gw_table["actual"].sum()))
    add("Totals add up", max(sums) - min(sums) < 1e-6 and max(acts) - min(acts) < 1e-9,
        f"predicted {sums[0]:,.1f} and actual {acts[0]:,.0f} the same whether added by player, "
        "by gameweek or match by match")
    return checks


def forecast_path_check(table: pd.DataFrame, ev, gw: int | None = None) -> dict:
    """Do two independent code paths agree? Gameweek `gw` (default: the latest
    replayed) predicted again from the training table (backtest.walk_forward: one
    row per match played) and compared, player by fixture, with the harness's
    forecast-path predictions."""
    from fpl_optimiser.model import backtest

    gw = gw or ev.rounds[-1]
    data = events.season_data(ev.season)
    wf = backtest.walk_forward(table, ev.season, rounds=[gw], events_data=data, half_life=ev.half_life)
    fw = ev.first_week[ev.first_week["round"] == gw]
    m = fw[["id", "opponent_team", "xp", "xp_events"]].merge(
        wf[["id", "opponent_team", "xp", "xp_events"]], on=["id", "opponent_team"], suffixes=("_h", "_t"))
    gap = float(max((m["xp_h"] - m["xp_t"]).abs().max(), (m["xp_events_h"] - m["xp_events_t"]).abs().max())) \
        if len(m) else float("nan")
    ok = len(m) == len(fw) == len(wf) and gap < 1e-9
    return {"check": "Two code paths give the same predictions", "ok": bool(ok),
            "detail": f"GW{gw} predicted from the training table and from the forecasting code: {len(m)} of "
                      f"{len(fw)} player-fixtures matched, largest difference {gap:.1e} points"}
