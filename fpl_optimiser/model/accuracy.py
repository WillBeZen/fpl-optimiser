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


def half_life_check(season: str, rounds: list[int], ids, half_lives, events_data: dict | None = None,
                    base_preds: pd.DataFrame | None = None,
                    base_half_life: float = config.RECENCY_HALF_LIFE_GW) -> pd.DataFrame:
    """Replay the same window at each half-life (rebuilding every form feature and the
    event model's rates with it) and score the same players. `base_preds` (the
    replay at base_half_life) is reused rather than recomputed."""
    from fpl_optimiser.features import multi_season
    from fpl_optimiser.model import backtest

    rows = []
    for h in half_lives:
        if base_preds is not None and h == base_half_life:
            preds = base_preds
        else:
            table = multi_season.build_all(half_life=h)
            preds = backtest.walk_forward(table, season, rounds, events_data, half_life=h)
        s = summary(player_gameweeks(preds, rounds, ids), h)
        rows.append({"half_life": h, "current": h == base_half_life, **s})
    return pd.DataFrame(rows)


def reconcile(pg: pd.DataFrame, preds: pd.DataFrame, rounds: list[int], season, ids,
              half_life: float, event_weight: float, plan_candidates: int | None = None,
              gw_table: pd.DataFrame | None = None) -> list[dict]:
    """Checks that the numbers are right: each {check, ok (True/False/None = can't
    tell), detail}. `season` is the filed Season (live points, pending matches,
    season totals); `preds` the replay's raw rows."""
    checks = []

    def add(check, ok, detail):
        checks.append({"check": check, "ok": None if ok is None else bool(ok), "detail": detail})

    gws = season.gameweeks.set_index("id")
    done = [g for g in rounds if bool(gws.at[g, "finished"])]
    checked = [g for g in rounds if bool(gws.get("data_checked", gws["finished"]).get(g, False))]
    add("The window is complete", len(done) == len(rounds) == len(checked),
        f"GW{rounds[0]}-{rounds[-1]}: {len(done)} of {len(rounds)} finished, {len(checked)} checked by FPL"
        if rounds else "no finished gameweeks yet")

    live = season.live
    have = sorted(set(live["gw"]) & set(rounds)) if len(live) else []
    if not have:
        add("Actual points match FPL's official gameweek totals", None,
            "FPL's gameweek totals aren't filed yet (they're filed by a data pull)")
    else:
        lv = live[live["gw"].isin(have) & live["id"].isin(set(ids))].rename(columns={"gw": "round"})
        m = lv[["id", "round", "total_points", "minutes"]].merge(
            pg[["id", "round", "actual", "minutes"]], on=["id", "round"], how="outer", suffixes=("_fpl", "_ours"))
        m = m[m["round"].isin(have)]
        # a player with no match record that gameweek scored 0 for us
        m[["actual", "minutes_ours"]] = m[["actual", "minutes_ours"]].fillna(0)
        m[["total_points", "minutes_fpl"]] = m[["total_points", "minutes_fpl"]].fillna(0)
        bad = m[(m["total_points"] != m["actual"]) | (m["minutes_fpl"] != m["minutes_ours"])]
        add("Actual points match FPL's official gameweek totals", bad.empty,
            f"{len(m) - len(bad)} of {len(m)} player-gameweeks identical (points and minutes) in "
            f"GW{have[0]}-{have[-1]}" + ("" if bad.empty else f"; {len(bad)} differ"))
        # the whole season so far, not just the window: every finished gameweek filed
        allgw = sorted(set(live["gw"]))
        ours = (season.history[season.history["round"].isin(allgw) & season.history["id"].isin(set(ids))]
                .groupby("id")["total_points"].sum())
        theirs = live[live["gw"].isin(allgw) & live["id"].isin(set(ids))].groupby("id")["total_points"].sum()
        both = pd.concat([ours.rename("ours"), theirs.rename("fpl")], axis=1).fillna(0)
        same = int((both["ours"] == both["fpl"]).sum())
        add("Every finished gameweek this season matches FPL", same == len(both),
            f"{same} of {len(both)} players' points over GW{allgw[0]}-{allgw[-1]} equal FPL's official totals")
        expl = live[live["gw"].isin(allgw)]
        add("FPL's own breakdown adds up", bool(expl["explained"].all()),
            f"each player's points by source sum to his total in {int(expl['explained'].sum())} of {len(expl)} cases")

    # FPL's season totals also count matches still being played; the filed match
    # records catch up at the next refresh. Finished gameweeks are unaffected.
    tot = pd.concat([season.history, season.pending])
    mine = tot[tot["id"].isin(set(ids))].groupby("id")["total_points"].sum()
    fpl_tot = season.players.set_index("id")["total_points"].reindex(mine.index)
    behind = int((mine != fpl_tot).sum())
    add("Match records are up to date", True if behind == 0 else None,
        f"all {len(mine)} players' records add up to FPL's season totals" if behind == 0 else
        f"{behind} of {len(mine)} players have points FPL already counts live (a gameweek in play) that "
        "the filed match records don't have yet; they arrive with the next refresh, and finished "
        "gameweeks are unaffected")

    late = pg[pg["trained_to"] >= pg["round"]]
    add("No prediction used later data", late.empty,
        "every gameweek was predicted from a model fitted only on earlier gameweeks"
        if late.empty else f"{len(late)} predictions trained on their own gameweek or later")

    win_all = preds[preds["round"].isin(rounds)]
    used = set()
    for col in ("half_life", "events_half_life"):          # recorded by build_all / walk_forward
        if col in win_all:
            used |= set(win_all[col].dropna().astype(float))
    used = used or {float(half_life)}
    add("Predictions use the half-life in config.py", used == {float(config.RECENCY_HALF_LIFE_GW)},
        "form features and event-model rates built with a half-life of "
        + ", ".join(f"{h:g}" for h in sorted(used)) + f" gameweeks (config.py: {config.RECENCY_HALF_LIFE_GW:g})")

    win = preds[preds["round"].isin(rounds) & preds["id"].isin(set(ids))]
    if "xp_blend" in win:
        gap = (win["xp_blend"] - ((1 - event_weight) * win["xp"] + event_weight * win["xp_events"])).abs().max()
        missing = int(win["xp_blend"].isna().sum())
        add("Predictions are the planner's blend", missing == 0 and (np.isnan(gap) or gap < 1e-9),
            f"(1 - {event_weight:g}) x points model + {event_weight:g} x event model for every prediction"
            + (f"; {missing} missing" if missing else ""))
    else:
        add("Predictions are the planner's blend", None, "the event model wasn't replayed (points model only)")

    if plan_candidates is not None:
        add("Players are the planner's candidate pool", len(set(ids)) == plan_candidates,
            f"{len(set(ids))} players here, {plan_candidates} in the planner's pool")

    sums = [float(pg["predicted"].sum()), float(win[predicted_column(win)].sum())]
    acts = [float(pg["actual"].sum()), float(win["target_points"].sum())]
    if gw_table is not None and len(gw_table):
        sums.append(float(gw_table["predicted"].sum()))
        acts.append(float(gw_table["actual"].sum()))
    add("Totals add up", max(sums) - min(sums) < 1e-6 and max(acts) - min(acts) < 1e-9,
        f"predicted {sums[0]:,.1f} and actual {acts[0]:,.0f} the same whether added by player, "
        "by gameweek or match by match")
    return checks


def forecast_path_check(table: pd.DataFrame, season_name: str, gw: int, preds: pd.DataFrame,
                        event_weight: float = config.EVENT_MODEL_WEIGHT) -> dict:
    """Is the replay exactly what the live forecasting code would have said? Rebuilds
    gameweek `gw` through predict's own path (features as of its deadline, the models
    fitted on earlier gameweeks, the event model, the blend), with that week's prices
    and no injury flags, and compares every player with the replay."""
    from fpl_optimiser.features import build, multi_season
    from fpl_optimiser.ingest import season_store
    from fpl_optimiser.model import points

    cur, prior, profiles = multi_season.season_inputs(season_name)
    players = cur.players
    if "value" in cur.history:                    # each player's price at the time, as the replay uses
        then = cur.history[cur.history["round"] == gw].groupby("id")["value"].first() / 10
        players = players.assign(price=players["id"].map(then).fillna(players["price"]))
    upcoming = cur.fixtures.assign(finished=cur.fixtures["finished"].where(cur.fixtures["gameweek"] != gw, False))
    rows = build.build_prediction_table(cur.history, upcoming, players, profiles, as_of_round=gw,
                                        gameweeks=[gw], prior=prior)
    known = table.dropna(subset=["target_points", "minutes"])
    train = known[(known["season"] < season_name) | ((known["season"] == season_name) & (known["round"] < gw))]
    out = rows.join(points.fit(train).predict(rows))
    prev_name = season_store.previous_season(season_name)
    prev = season_store.load(prev_name) if prev_name in season_store.available() else None
    ratings = events.team_ratings(cur.fixtures, cur.teams, gw, cur.history,
                                  prev.fixtures if prev else None, prev.teams if prev else None)
    rates = events.player_rates(cur.history, cur.players, gw, prior)
    out["xp_events"] = events.expected_events(out, ratings, rates)["xp_events"].to_numpy()
    path_xp = ((1 - event_weight) * out["xp"] + event_weight * out["xp_events"]).groupby(out["id"]).sum()
    col = predicted_column(preds)
    replay = preds[preds["round"] == gw].groupby("id")[col].sum()
    both = pd.concat([path_xp.rename("path"), replay.rename("replay")], axis=1).dropna()
    gap = float((both["path"] - both["replay"]).abs().max()) if len(both) else float("nan")
    ok = len(both) > 0 and gap < 1e-6
    return {"check": "The replay is exactly what the forecast would have said", "ok": bool(ok),
            "detail": f"GW{gw} rebuilt through the live forecasting code for {len(both)} players: "
                      f"largest difference {gap:.1e} points (same prices as then, no injury flags)"}
