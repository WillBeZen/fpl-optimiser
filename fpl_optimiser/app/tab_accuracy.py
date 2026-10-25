"""Tab · Accuracy — how predicted points lined up with what players actually scored.

The last config.ACCURACY_WINDOW_GW finished gameweeks, replayed (model/accuracy.py):
each gameweek predicted by models fitted only on what came before it, with the
half-life in config.py and the planner's blend, then compared with FPL's points.
Players: the planner's candidate pool (everyone not dominated, your squad included).

  summary        headline numbers, the pool against every player
  gameweeks      predicted vs actual per gameweek, with the half-life's weights
  every player   each pool player's predicted and actual points, gameweek by gameweek
  player totals  predicted vs actual over the window, as a chart
  calibration    when the model predicts X, what happens
  breakdown      by position, and by scoring source (against FPL's own breakdown)
  misses         the biggest misses, and how much came from players who didn't play
  half-life      the same window replayed at other half-lives
  as logged      the forecasts actually made at each deadline (from GW9 on)
  reconciliation every check that the numbers are right
"""
from __future__ import annotations

import pandas as pd

import config
from fpl_optimiser.app import charts
from fpl_optimiser.app.common import Context, WeekResults, money, pct, player_info
from fpl_optimiser.app.html import (Report, Section, card, data_table, error_class, esc, html_table, kv,
                                    note, pill)
from fpl_optimiser.model import accuracy, forecast_log
from fpl_optimiser.optimise import pool

TAB = "accuracy"
METRIC_FMT = {"Predicted": "{:,.1f}", "Actual": "{:,.0f}", "Bias": "{:+.2f}", "Typical miss (MAE)": "{:.2f}",
              "RMSE": "{:.2f}", "Rank correlation": "{:.2f}", "Within 2 points": pct, "Top 11 captured": pct}


def _metrics_row(label: str, s: dict) -> dict:
    return {"": label, "Player-gameweeks": s["player_gameweeks"], "Predicted": s["predicted"],
            "Actual": s["actual"], "Bias": s["bias"], "Typical miss (MAE)": s["mae"], "RMSE": s["rmse"],
            "Rank correlation": s["rank_corr"], "Within 2 points": s["within_2"], "Top 11 captured": s["captured"]}


def _status(ok) -> str:
    return pill("✓ pass", "good") if ok is True else (pill("✗ fail", "bad") if ok is False else pill("– info", "warn"))


def checks(w: WeekResults, ctx: Context, rounds: list[int], ids, pg: pd.DataFrame,
           gwt: pd.DataFrame | None = None) -> list[dict]:
    """The reconciliation checklist for this report (also printed by run_week)."""
    fc = w.forecast
    out = accuracy.reconcile(pg, w.replay, rounds, ctx.season, ids, config.RECENCY_HALF_LIFE_GW,
                             fc.event_weight, w.plan.stats.get("candidates") if w.plan is not None else None, gwt)
    if w.forecast_check:
        out.insert(out.index(next(c for c in out if c["check"] == "No prediction used later data")) + 1,
                   w.forecast_check)
    return out


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    fc = w.forecast
    if w.replay is None or w.replay.empty or ctx is None or ctx.season is None:
        r.sections.append(Section("acc-summary", "How accurate was the model?",
                                  "", [note("Not run this time (--quick). Run python scripts/run_week.py "
                                            "without --quick to replay the last gameweeks.")], tab=TAB))
        return
    season = ctx.season
    rounds = [g for g in accuracy.window(season.gameweeks) if g in set(w.replay["round"])]
    if not rounds:
        r.sections.append(Section("acc-summary", "How accurate was the model?", "",
                                  [note("No finished gameweeks to check yet.")], tab=TAB))
        return
    keep = list(w.team.squad_ids) if w.team is not None else []
    ids = pool.prune(fc.xp, fc.players, keep=keep)
    mine = set(keep)
    info = player_info(w, ctx)
    pg = accuracy.player_gameweeks(w.replay, rounds, ids)
    allpg = accuracy.player_gameweeks(w.replay, rounds)
    s, s_all = accuracy.summary(pg), accuracy.summary(allpg)
    gwt = accuracy.by_gameweek(pg)
    recon = checks(w, ctx, rounds, ids, pg, gwt)
    passed = sum(c["ok"] is True for c in recon)
    failed = sum(c["ok"] is False for c in recon)
    span = f"GW{rounds[0]}-GW{rounds[-1]}"
    h = config.RECENCY_HALF_LIFE_GW

    # 1 · summary --------------------------------------------------------------------
    comp = pd.DataFrame([_metrics_row(f"Pool ({s['players']} players)", s),
                         _metrics_row(f"Every player ({s_all['players']})", s_all)])
    r.sections.append(Section(
        "acc-summary", f"How accurate was the model? {span}",
        f"The last {len(rounds)} finished gameweeks, replayed: each predicted by models fitted only on what "
        f"came before it, with config.py's half-life ({h:g} gameweeks) in every recency weight and the "
        f"planner's blend ({1 - fc.event_weight:g} points model + {fc.event_weight:g} event model), then "
        "compared with what FPL scored. Players: the planner's candidate pool, everyone not dominated "
        f"({len(ids)}, your squad included).",
        [card(kv([("Window", span), ("Players", str(s["players"])),
                  ("Player-gameweeks", f"{s['player_gameweeks']:,}"), ("Predicted", f"{s['predicted']:,.0f}"),
                  ("Actual", f"{s['actual']:,.0f}"), ("Bias per player-gameweek", f"{s['bias']:+.2f}"),
                  ("Typical miss", f"{s['mae']:.2f} pts"), ("Within 2 points", pct(s["within_2"])),
                  ("Rank correlation", f"{s['rank_corr']:.2f}"), ("Top 11 captured", pct(s["captured"])),
                  ("Reconciliation", f"{passed} of {len(recon)} checks pass" + (f", {failed} fail" if failed else ""))]),
              data_table(comp, fmt=METRIC_FMT, sortable=False),
              note("Bias = predicted minus actual, so negative means players scored more than predicted. "
                   "The pool is chosen for being good now, which favours players who have been doing well: "
                   "it under-predicts them a little, while across every player the model is nearly unbiased. "
                   "Top 11 captured = the real points of the 11 it rated highest each gameweek, as a share of "
                   "the best 11 in hindsight. FPL's injury flags at past deadlines aren't archived, so a "
                   "player ruled out at the time still has a prediction here."))], tab=TAB))

    # 2 · gameweek by gameweek ----------------------------------------------------------
    gtab = gwt.rename(columns={"weight": "Half-life weight", "players": "Players", "predicted": "Predicted",
                               "actual": "Actual", "difference": "Predicted - actual", "bias": "Bias",
                               "mae": "Typical miss", "rmse": "RMSE", "rank_corr": "Rank correlation",
                               "top11": "Top 11 (model)", "best11": "Best 11 (hindsight)",
                               "captured": "Captured"})
    r.sections.append(Section(
        "acc-gameweeks", "Gameweek by gameweek",
        f"Totals over the pool each gameweek. The half-life weight is how much each gameweek counts in "
        f"the model's form today (the latest 1, then halving every {h:g} gameweeks); weighted that way, the "
        f"typical miss is {s['weighted_mae']:.2f} points and the bias {s['weighted_bias']:+.2f}.",
        [card(r.chart("acc-gameweeks", charts.accuracy_bars, gwt),
              data_table(gtab, fmt={"Half-life weight": "{:.2f}", "Predicted": "{:,.1f}", "Actual": "{:,.0f}",
                                    "Predicted - actual": "{:+,.1f}", "Bias": "{:+.2f}", "Typical miss": "{:.2f}",
                                    "RMSE": "{:.2f}", "Rank correlation": "{:.2f}", "Top 11 (model)": "{:.0f}",
                                    "Best 11 (hindsight)": "{:.0f}", "Captured": pct}, sortable=False))],
        tab=TAB))

    # 3 · every player --------------------------------------------------------------------
    by = {(i, g): row for i, g, row in zip(pg["id"], pg["round"], pg.itertuples())}
    rows = []
    for pid, d in pg.groupby("id"):
        p = info.loc[pid] if pid in info.index else None
        row = {"Player": p["web_name"] if p is not None else str(pid),
               "Club": (p["team_short"] if p is not None and isinstance(p["team_short"], str) else
                        (p["team_name"] if p is not None else "")),
               "Pos": p["position"] if p is not None else "", "Price": p["price"] if p is not None else None,
               "Yours": "yes" if pid in mine else ""}
        for g in rounds:
            x = by.get((pid, g))
            if x is None:
                row[f"GW{g}"] = (None, "–")
            else:
                row[f"GW{g}"] = (float(x.actual), f"{x.actual:g}<small>pred {x.predicted:.1f}</small>",
                                 error_class(x.error))
        row["Predicted"] = float(d["predicted"].sum())
        row["Actual"] = float(d["actual"].sum())
        row["Predicted - actual"] = row["Predicted"] - row["Actual"]
        row["Typical miss"] = float(d["error"].abs().mean())
        row["Minutes"] = int(d["minutes"].sum())
        row["_id"] = pid
        rows.append(row)
    every = pd.DataFrame(rows).sort_values("Actual", ascending=False)
    gw_cols = [f"GW{g}" for g in rounds]
    legend = ('<div class="legend">Each cell: points scored, with the prediction under it. '
              '<span class="e0">scored far more</span><span class="e2">a bit more</span>'
              '<span class="e3">about right</span><span class="e4">a bit less</span>'
              '<span class="e6">far less</span></div>')
    clubs = sorted(every["Club"].dropna().unique())
    filters = ('<div class="filters" data-table="acc-table">'
               '<input type="search" data-filter="text" placeholder="Search player or club" aria-label="Search">'
               '<label>Position <select data-filter="pos"><option value="">All</option>'
               + "".join(f'<option value="{p}">{p}</option>' for p in config.POSITION_QUOTA)
               + '</select></label><label>Club <select data-filter="club"><option value="">All</option>'
               + "".join(f'<option value="{esc(c.lower())}">{esc(c)}</option>' for c in clubs)
               + '</select></label><span class="count"></span></div>')
    tbl = data_table(every, id="acc-table", hidden=("_id",), html_cols=tuple(gw_cols),
                     fmt={"Price": money, "Predicted": "{:.1f}", "Actual": "{:.0f}",
                          "Predicted - actual": "{:+.1f}", "Typical miss": "{:.2f}"},
                     tall=True, row_attrs=lambda row: {"pos": row["Pos"], "club": str(row["Club"]).lower(),
                                                       **({"class": "mine"} if row["_id"] in mine else {})})
    r.sections.append(Section(
        "acc-players", "Every player in the pool",
        f"All {len(every)} players the planner considers, gameweek by gameweek. Sorted by points scored; click "
        "a column to sort (a gameweek column sorts by points). A dash: no match that gameweek.",
        [card(legend, filters, tbl)], tab=TAB))

    # 4 · player totals chart -----------------------------------------------------------------
    tot = every.assign(web_name=every["Player"], team_name=every["Club"], position=every["Pos"],
                       predicted=every["Predicted"], actual=every["Actual"], yours=every["_id"].isin(mine))
    r.sections.append(Section(
        "acc-scatter", "Player by player",
        f"Each player's total over {span}: predicted (across) against scored (up). Above the dotted line he "
        "scored more than predicted; below it, less. The furthest from the line are labelled.",
        [card(r.chart("acc-scatter", charts.accuracy_scatter, tot))], tab=TAB))

    # 5 · calibration ---------------------------------------------------------------------
    cal = accuracy.calibration(pg)
    ctab = cal.rename(columns={"bin": "Predicted", "n": "Player-gameweeks", "predicted": "Average predicted",
                               "actual": "Average scored", "returned": "Scored 4+"})
    r.sections.append(Section(
        "acc-calibration", "Calibration",
        "Every player-gameweek grouped by its prediction. If the model is calibrated, players it gave 3-4 "
        "points averaged about 3.5. Points left of the dotted line: the model was too cautious; right of it, "
        "too optimistic.",
        [card(r.chart("acc-calibration", charts.calibration_chart, cal),
              data_table(ctab, fmt={"Average predicted": "{:.2f}", "Average scored": "{:.2f}", "Scored 4+": pct},
                         sortable=False))], tab=TAB))

    # 6 · position and source ---------------------------------------------------------------
    pos = accuracy.by_position(pg, fc.players.set_index("id")["position"]).rename(
        columns={"position": "Position", "player_gameweeks": "Player-gameweeks", "predicted": "Average predicted",
                 "actual": "Average scored", "bias": "Bias", "mae": "Typical miss", "rmse": "RMSE"})
    blocks = [f"<h3>By position</h3>" + data_table(pos, fmt={"Average predicted": "{:.2f}", "Average scored": "{:.2f}",
                                                           "Bias": "{:+.2f}", "Typical miss": "{:.2f}", "RMSE": "{:.2f}"},
                                                 sortable=False)]
    src = accuracy.by_source(pg, season.live)
    if len(src):
        src_tbl = src.rename(columns={"source": "Source", "predicted": "Event model predicted",
                                      "actual": "FPL actual", "difference": "Predicted - actual"})
        blocks.append("<h3>By scoring source</h3>" + data_table(
            src_tbl, fmt={"Event model predicted": "{:,.1f}", "FPL actual": "{:,.0f}", "Predicted - actual": "{:+,.1f}"},
            sortable=False))
    r.sections.append(Section(
        "acc-breakdown", "By position and by scoring source",
        "Where the misses come from. The source split compares the event model (the half of the blend that "
        "works in goals, assists, clean sheets...) with FPL's own breakdown of the points scored.",
        [card(*blocks, pair=len(blocks) == 2)], tab=TAB))

    # 7 · misses -----------------------------------------------------------------------------
    ms = accuracy.minutes_split(pg)
    over, under = accuracy.misses(pg)
    def miss_table(d: pd.DataFrame) -> str:
        t = pd.DataFrame({"Player": d["id"].map(info["web_name"]), "Club": d["id"].map(info["team_short"]),
                          "GW": d["round"], "Predicted": d["predicted"], "Scored": d["actual"],
                          "Minutes": d["minutes"], "P(plays) predicted": d["p_play"], "Why": d["reason"]})
        return data_table(t, fmt={"Predicted": "{:.1f}", "Scored": "{:.0f}", "P(plays) predicted": pct})
    r.sections.append(Section(
        "acc-misses", "Biggest misses",
        f"{ms['didnt_play']} of the {ms['didnt_play'] + ms['played']} player-gameweeks were players who didn't "
        f"play at all; they account for {ms['share_from_absent']:.0%} of the total miss. The rest is the "
        "football: goals, assists, clean sheets and bonus that no model sees coming.",
        [card(f"<h3>Scored far less than predicted</h3>{miss_table(over)}",
              f"<h3>Scored far more than predicted</h3>{miss_table(under)}", pair=True)], tab=TAB))

    # 8 · half-life check ----------------------------------------------------------------------
    if w.half_lives is not None and len(w.half_lives):
        hl = w.half_lives.assign(label=[f"{x:g}" + (" (config.py)" if c else "")
                                        for x, c in zip(w.half_lives["half_life"], w.half_lives["current"])])
        htab = pd.DataFrame({"Half-life (gameweeks)": hl["label"], "Bias": hl["bias"], "Typical miss": hl["mae"],
                             "RMSE": hl["rmse"], "Rank correlation": hl["rank_corr"],
                             "Top 11 (model)": hl["top11"], "Top 11 captured": hl["captured"],
                             "Weighted typical miss": hl["weighted_mae"]})
        best = hl.loc[hl["rmse"].idxmin(), "half_life"]
        msg = (f"The current half-life ({h:g}) has the lowest RMSE of the three." if best == h else
               f"A half-life of {best:g} had the lowest RMSE on this window. One window is a small sample, so "
               "check it over a few weeks before changing RECENCY_HALF_LIFE_GW in config.py.")
        blocks = [data_table(htab, fmt={"Bias": "{:+.2f}", "Typical miss": "{:.3f}", "RMSE": "{:.3f}",
                                        "Rank correlation": "{:.3f}", "Top 11 (model)": "{:.1f}",
                                        "Top 11 captured": pct, "Weighted typical miss": "{:.3f}"}, sortable=False),
                  note(msg)]
    else:
        blocks = [note("Not run this time (--quick).")]
    r.sections.append(Section(
        "acc-halflife", "Half-life check",
        f"The same {span} replayed with a shorter and a longer half-life (every form feature and the event "
        "model's rates rebuilt), scored on the same players. A shorter half-life chases recent form; a longer "
        "one trusts the longer record.", [card(*blocks)], tab=TAB))

    # 9 · as logged at the deadline --------------------------------------------------------------
    logged = forecast_log.load(season.season, rounds)
    if len(logged):
        lg = logged[logged["id"].isin(ids)].rename(columns={"gw": "round"})
        m = lg.merge(pg[["id", "round", "actual"]], on=["id", "round"], how="inner")
        rows = []
        for g, d in m.groupby("round"):
            e = d["xp"] - d["actual"]
            rows.append({"GW": int(g), "Logged at": d["logged_at"].iloc[0], "Players": len(d),
                         "Predicted": d["xp"].sum(), "Actual": d["actual"].sum(), "Bias": e.mean(),
                         "Typical miss": e.abs().mean()})
        blocks = [data_table(pd.DataFrame(rows), fmt={"Predicted": "{:,.1f}", "Actual": "{:,.0f}",
                                                      "Bias": "{:+.2f}", "Typical miss": "{:.2f}"}, sortable=False)]
    else:
        blocks = [note(f"Each weekly run now files its forecast for the next gameweek, injury flags and all "
                       f"(data/processed/forecast_log/). From GW{fc.next_gw}, once it's played, this section "
                       "compares the forecasts actually made at each deadline with what happened.")]
    r.sections.append(Section("acc-logged", "As forecast at the deadline", "", [card(*blocks)], tab=TAB))

    # 10 · reconciliation --------------------------------------------------------------------------
    r.sections.append(Section(
        "acc-reconcile", "Reconciliation",
        "Every check that the numbers on this tab are right. 'Info' marks something worth knowing that "
        "doesn't affect the results.",
        [card(html_table(["Check", "Result", "Detail"],
                         [[esc(c["check"]), _status(c["ok"]), f'<div class="cell-wrap">{esc(c["detail"])}</div>']
                          for c in recon]))], tab=TAB))
