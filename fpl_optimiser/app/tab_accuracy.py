"""Tab · Accuracy — how predicted points lined up with what players actually scored,
measured honestly (model/evaluation.py).

Every finished gameweek is replayed as it stood at its deadline: models fitted only
on earlier gameweeks, features and the player pool as of that deadline, the blend's
weight chosen from earlier weeks only. The planner's blend is then scored against
simple methods anyone could use, on exactly the same player-gameweeks, with the
gameweeks as the unit of evidence.

  summary        every method side by side; is the model better than simple methods?
  gameweeks      each method gameweek by gameweek, and the blend's weight each week
  every player   the last few gameweeks, player by player (predicted vs scored)
  player totals  the same, as a chart
  calibration    when the model predicts X, what happens
  breakdown      by position, and by scoring source against FPL's own breakdown
  misses         the biggest misses
  half-life      the same deadlines replayed with a shorter and a longer half-life
  as logged      the forecasts actually made at each deadline (from GW9 on)
  data checks    that the numbers are right (separate from how good the model is)
"""
from __future__ import annotations

import pandas as pd

import config
from fpl_optimiser.app import charts
from fpl_optimiser.app.common import Context, WeekResults, money, pct, player_info
from fpl_optimiser.app.html import (Report, Section, card, data_table, error_class, esc, html_table, kv,
                                    note, pill)
from fpl_optimiser.model import accuracy, evaluation, forecast_log

TAB = "accuracy"
SHOWN = ["xp_blend", "xp", "pos_price_p", "season_avg"]       # the chart's four lines
METRIC_NAMES = {"rmse": "RMSE (accuracy)", "rank_corr": "Rank correlation (ordering)",
                "top11": "Top 11 points (picking)"}


def _status(ok) -> str:
    return pill("✓ pass", "good") if ok is True else (pill("✗ fail", "bad") if ok is False else pill("– info", "warn"))


def _verdict(metric: str, low: float, high: float) -> str:
    """Is the reference clearly better (the 90% interval of its advantage excludes 0)?"""
    if pd.isna(low) or pd.isna(high):
        return "Too few gameweeks"
    better = (high < 0) if metric == "rmse" else (low > 0)
    worse = (low > 0) if metric == "rmse" else (high < 0)
    return "Blend better" if better else ("This method better" if worse else "No clear difference")


def checks(w: WeekResults, ctx: Context, ev, rounds: list[int], pg: pd.DataFrame | None = None,
           gw_table: pd.DataFrame | None = None) -> list[dict]:
    """The data checks for this report (also printed by run_week)."""
    out = accuracy.reconcile(ev, ctx.season, rounds, pg, gw_table)
    if w.forecast_check:
        out.insert(next((i + 1 for i, c in enumerate(out) if c["check"] == "No prediction used later data"),
                        len(out)), w.forecast_check)
    return out


def _method_table(s: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Method": s["label"], "Player-gameweeks": s["n"], "Bias": s["bias"],
                         "Typical miss": s["mae"], "RMSE": s["rmse"], "Rank correlation": s["rank_corr"],
                         "Top 11 a gameweek": s["top11"], "Top 11 spread (SD)": s["top11_sd"],
                         "Top 11 captured": s["captured"]})


METHOD_FMT = {"Bias": "{:+.2f}", "Typical miss": "{:.2f}", "RMSE": "{:.3f}", "Rank correlation": "{:.3f}",
              "Top 11 a gameweek": "{:.1f}", "Top 11 spread (SD)": "{:.1f}", "Top 11 captured": pct}


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    fc = w.forecast
    ev = w.evaluation
    if ev is None or ev.first_week.empty or ctx is None or ctx.season is None:
        r.sections.append(Section("acc-summary", "How accurate was the model?", "",
                                  [note("Not run this time (--quick). Run python scripts/run_week.py without "
                                        "--quick to replay past gameweeks.")], tab=TAB))
        return
    season = ctx.season
    rounds = ev.rounds
    span = f"GW{rounds[0]}-GW{rounds[-1]}"
    info = player_info(w, ctx)
    mine = set(w.team.squad_ids) if w.team is not None else set()
    h = ev.half_life

    pgw = {pop: evaluation.player_gameweeks(ev, pop) for pop in evaluation.POPULATIONS}
    gws = {pop: evaluation.by_gameweek(pgw[pop]) for pop in pgw}
    summ = {pop: evaluation.summary(gws[pop], pgw[pop]) for pop in pgw}
    pairs = evaluation.paired(gws["pool"])
    pg = accuracy.reference_rows(ev, rounds, "pool")
    ref_gw = accuracy.by_gameweek(pg, h)
    recon = checks(w, ctx, ev, rounds, pg, ref_gw)
    failed = [c for c in recon if c["ok"] is False]
    s = accuracy.summary(pg, h)

    # 1 · summary ------------------------------------------------------------------------
    ptab = pd.DataFrame({
        "Method": pairs["label"], "Measure": pairs["metric"].map(METRIC_NAMES),
        "Blend minus method": pairs["mean_diff"],
        "90% interval": [f"{lo:+.3f} to {hi:+.3f}" if pd.notna(lo) else "–" for lo, hi in zip(pairs["low"], pairs["high"])],
        "Blend better in": [f"{b} of {g} GWs" for b, g in zip(pairs["reference_better"], pairs["gameweeks"])],
        "Verdict": [_verdict(m, lo, hi) for m, lo, hi in zip(pairs["metric"], pairs["low"], pairs["high"])]})
    r.sections.append(Section(
        "acc-summary", f"How accurate was the model? {span}",
        f"Every finished gameweek since the first, replayed as it stood at its deadline: models fitted only on "
        f"earlier gameweeks, the players and their prices then, the planner's pool rebuilt from that deadline's "
        f"own forecast, and the blend's weight chosen only from earlier weeks. Every method is scored on the same "
        f"player-gameweeks. The half-life is config.py's ({h:g} gameweeks).",
        [card(kv([("Gameweeks", f"{len(rounds)} ({span})"), ("Pool player-gameweeks", f"{s['player_gameweeks']:,}"),
                  ("Predicted", f"{s['predicted']:,.0f}"), ("Scored", f"{s['actual']:,.0f}"),
                  ("Bias", f"{s['bias']:+.2f}"), ("Typical miss", f"{s['mae']:.2f} pts"),
                  ("Rank correlation", f"{s['rank_corr']:.2f}"), ("Top 11 captured", pct(s["captured"])),
                  ("Data checks", "all pass" if not failed else f"{len(failed)} fail")]),
              "<h3>Every method, the planner's pool at each deadline</h3>",
              data_table(_method_table(summ["pool"]), fmt=METHOD_FMT, sortable=False),
              "<h3>Every method, every player</h3>",
              data_table(_method_table(summ["all"]), fmt=METHOD_FMT, sortable=False),
              note("RMSE and typical miss: how far off each prediction was (lower is better). Rank correlation: "
                   "how well it ordered players within a gameweek. Top 11: the real points of the 11 players it "
                   "rated highest each gameweek, the measure closest to picking a team. 'Blend, config weight' "
                   "uses the 0.5 that was chosen by looking at these same weeks, so it is optimistic; the "
                   "reference ('weight from earlier weeks') is not. Position/price x P(plays) uses the minutes "
                   "model, so it isn't model-free; Position/price is."),
              "<h3>Is the blend better than each simple method?</h3>",
              data_table(ptab, fmt={"Blend minus method": "{:+.3f}"}, sortable=False),
              note(f"Gameweek by gameweek, paired: the gameweeks are the independent evidence (hundreds of "
                   f"players share each gameweek's fixtures), so with {len(rounds)} of them the 90% intervals are "
                   "wide. 'No clear difference' means this many gameweeks can't separate the two. For RMSE a "
                   "negative difference favours the blend; for the others a positive one."))], tab=TAB))

    # 2 · gameweek by gameweek ---------------------------------------------------------------
    labels = evaluation.METHODS
    shown = [m for m in SHOWN if m in set(gws["pool"]["method"])]

    def pivot(metric):
        p = gws["pool"].pivot(index="round", columns="method", values=metric)
        p = p[[m for m in labels if m in p.columns]].rename(columns=labels)
        return p.reset_index().rename(columns={"round": "GW"})
    weights = pd.DataFrame({"GW": list(ev.weights), "Event-model weight": list(ev.weights.values()),
                            "Chosen from": [", ".join(f"GW{x}" for x in ev.weight_rounds.get(g, [])) or
                                            f"config.py ({config.EVENT_MODEL_WEIGHT:g}, nothing earlier)"
                                            for g in ev.weights]})
    r.sections.append(Section(
        "acc-gameweeks", "Gameweek by gameweek",
        "Each method's top 11 and RMSE in each gameweek (the planner's pool). Week-to-week swings are large, "
        "which is why a few gameweeks can't settle close comparisons.",
        [card(r.chart("acc-methods", charts.method_lines, gws["pool"], "top11", shown, labels,
                      "Real points of each method's top 11", "Points"),
              "<h3>Top 11 points</h3>", data_table(pivot("top11"), fmt={c: "{:.0f}" for c in labels.values()},
                                                   sortable=False),
              "<h3>RMSE</h3>", data_table(pivot("rmse"), fmt={c: "{:.3f}" for c in labels.values()}, sortable=False),
              "<h3>The blend's weight each gameweek</h3>",
              data_table(weights, fmt={"Event-model weight": "{:.2f}"}, sortable=False),
              note("The weight is the one with the lowest RMSE over all earlier gameweeks' predictions; 1.0 means "
                   "the event model alone.")),
         card(r.chart("acc-gameweeks", charts.accuracy_bars, ref_gw))], tab=TAB))

    # 3 · every player (the last few gameweeks) --------------------------------------------------
    lens = rounds[-config.ACCURACY_WINDOW_GW:]
    allpg = accuracy.reference_rows(ev, lens, "all")
    allpg["in_pool"] = [i in ev.pools.get(g, set()) for i, g in zip(allpg["id"], allpg["round"])]
    by = {(i, g): x for i, g, x in zip(allpg["id"], allpg["round"], allpg.itertuples())}
    ids = sorted({i for i, p in zip(allpg["id"], allpg["in_pool"]) if p})
    rows = []
    for pid in ids:
        p = info.loc[pid] if pid in info.index else None
        row = {"Player": p["web_name"] if p is not None else str(pid),
               "Club": p["team_short"] if p is not None and isinstance(p["team_short"], str) else "",
               "Pos": p["position"] if p is not None else "", "Price now": p["price"] if p is not None else None,
               "Yours": "yes" if pid in mine else ""}
        pred = act = minutes = 0.0
        errs, weeks = [], 0
        for g in lens:
            x = by.get((pid, g))
            if x is None:
                row[f"GW{g}"] = (None, "–")
            elif x.in_pool:
                row[f"GW{g}"] = (float(x.actual), f"{x.actual:g}<small>pred {x.predicted:.1f}</small>",
                                 error_class(x.error))
                pred, act, minutes, weeks = pred + x.predicted, act + x.actual, minutes + x.minutes, weeks + 1
                errs.append(abs(x.error))
            else:
                row[f"GW{g}"] = (float(x.actual), f"{x.actual:g}<small>not in pool</small>")
        row.update({"Weeks in pool": weeks, "Predicted": pred, "Scored": act, "Predicted - scored": pred - act,
                    "Typical miss": sum(errs) / len(errs) if errs else float("nan"), "Minutes": int(minutes),
                    "_id": pid})
        rows.append(row)
    every = pd.DataFrame(rows).sort_values("Scored", ascending=False) if rows else pd.DataFrame()
    gw_cols = [f"GW{g}" for g in lens]
    legend = ('<div class="legend">Each cell: points scored, with the prediction under it (in that '
              'deadline\'s pool). <span class="e0">scored far more</span><span class="e2">a bit more</span>'
              '<span class="e3">about right</span><span class="e4">a bit less</span>'
              '<span class="e6">far less</span></div>')
    clubs = sorted(every["Club"].dropna().unique()) if len(every) else []
    filters = ('<div class="filters" data-table="acc-table">'
               '<input type="search" data-filter="text" placeholder="Search player or club" aria-label="Search">'
               '<label>Position <select data-filter="pos"><option value="">All</option>'
               + "".join(f'<option value="{p}">{p}</option>' for p in config.POSITION_QUOTA)
               + '</select></label><label>Club <select data-filter="club"><option value="">All</option>'
               + "".join(f'<option value="{esc(c.lower())}">{esc(c)}</option>' for c in clubs)
               + '</select></label><span class="count"></span></div>')
    tbl = data_table(every, id="acc-table", hidden=("_id",), html_cols=tuple(gw_cols),
                     fmt={"Price now": money, "Predicted": "{:.1f}", "Scored": "{:.0f}",
                          "Predicted - scored": "{:+.1f}", "Typical miss": "{:.2f}"},
                     tall=True, row_attrs=lambda row: {"pos": row["Pos"], "club": str(row["Club"]).lower(),
                                                       **({"class": "mine"} if row["_id"] in mine else {})}) \
        if len(every) else note("No pool players in these gameweeks.")
    r.sections.append(Section(
        "acc-players", f"Every player, GW{lens[0]}-GW{lens[-1]}",
        f"Every player in the planner's pool at any of the last {len(lens)} deadlines (each deadline's pool, "
        f"rebuilt from its own forecast). Totals count only the weeks he was in the pool. Sorted by points scored.",
        [card(legend, filters, tbl)], tab=TAB))

    # 4 · player totals chart ------------------------------------------------------------------
    lens_pg = accuracy.reference_rows(ev, lens, "pool")
    tot = lens_pg.groupby("id")[["predicted", "actual"]].sum().reset_index()
    tot = tot.assign(web_name=tot["id"].map(info["web_name"]), team_name=tot["id"].map(info["team_short"]),
                     position=tot["id"].map(info["position"]), yours=tot["id"].isin(mine))
    r.sections.append(Section(
        "acc-scatter", "Player by player",
        f"Each player's total over his pool weeks in GW{lens[0]}-GW{lens[-1]}: predicted (across) against scored "
        "(up). Above the dotted line he scored more than predicted. The furthest from the line are labelled.",
        [card(r.chart("acc-scatter", charts.accuracy_scatter, tot))], tab=TAB))

    # 5 · calibration -------------------------------------------------------------------------
    cal = accuracy.calibration(pg)
    ctab = cal.rename(columns={"bin": "Predicted", "n": "Player-gameweeks", "predicted": "Average predicted",
                               "actual": "Average scored", "returned": "Scored 4+"})
    r.sections.append(Section(
        "acc-calibration", "Calibration",
        "Every pool player-gameweek grouped by its prediction. If the model is calibrated, players it gave 3-4 "
        "points averaged about 3.5. Above the dotted line: the model was too cautious; below it, too optimistic.",
        [card(r.chart("acc-calibration", charts.calibration_chart, cal),
              data_table(ctab, fmt={"Average predicted": "{:.2f}", "Average scored": "{:.2f}", "Scored 4+": pct},
                         sortable=False))], tab=TAB))

    # 6 · position and source -------------------------------------------------------------------
    pos = accuracy.by_position(pg, fc.players.set_index("id")["position"]).rename(
        columns={"position": "Position", "player_gameweeks": "Player-gameweeks", "predicted": "Average predicted",
                 "actual": "Average scored", "bias": "Bias", "mae": "Typical miss", "rmse": "RMSE"})
    blocks = ["<h3>By position</h3>" + data_table(pos, fmt={"Average predicted": "{:.2f}", "Average scored": "{:.2f}",
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
        "Where the misses come from (the pool, every replayed gameweek). The source split compares the event "
        "model's goals, assists, clean sheets... with FPL's own breakdown of the points scored.",
        [card(*blocks, pair=len(blocks) == 2)], tab=TAB))

    # 7 · misses ---------------------------------------------------------------------------------
    ms = accuracy.minutes_split(pg)
    over, under = accuracy.misses(pg)

    def miss_table(d: pd.DataFrame) -> str:
        t = pd.DataFrame({"Player": d["id"].map(info["web_name"]), "Club": d["id"].map(info["team_short"]),
                          "GW": d["round"], "Predicted": d["predicted"], "Scored": d["actual"],
                          "Minutes": d["minutes"], "P(plays) predicted": d["p_play"], "Why": d["reason"]})
        return data_table(t, fmt={"Predicted": "{:.1f}", "Scored": "{:.0f}", "P(plays) predicted": pct})
    r.sections.append(Section(
        "acc-misses", "Biggest misses",
        f"{ms['didnt_play']} of the {ms['didnt_play'] + ms['played']} pool player-gameweeks were players who "
        f"didn't play at all; they account for {ms['share_from_absent']:.0%} of the total miss. Injury flags "
        "at past deadlines aren't archived, so the replay predicts everyone as if fit.",
        [card(f"<h3>Scored far less than predicted</h3>{miss_table(over)}",
              f"<h3>Scored far more than predicted</h3>{miss_table(under)}", pair=True)], tab=TAB))

    # 8 · half-life check ---------------------------------------------------------------------------
    if w.half_lives is not None and len(w.half_lives):
        hl = w.half_lives
        htab = pd.DataFrame({"Half-life (gameweeks)": [f"{x:g}" + (" (config.py)" if c else "")
                                                      for x, c in zip(hl["half_life"], hl["current"])],
                             "Bias": hl["bias"], "Typical miss": hl["mae"], "RMSE": hl["rmse"],
                             "Rank correlation": hl["rank_corr"], "Top 11 a gameweek": hl["top11"],
                             "Top 11 captured": hl["captured"]})
        blocks = [data_table(htab, fmt={"Bias": "{:+.2f}", "Typical miss": "{:.3f}", "RMSE": "{:.3f}",
                                        "Rank correlation": "{:.3f}", "Top 11 a gameweek": "{:.1f}",
                                        "Top 11 captured": pct}, sortable=False),
                  note("Information only: choosing the half-life by looking at these same gameweeks and then "
                       "quoting them as evidence would be circular. A half-life is chosen on last season's "
                       "gameweeks (once they're filed in full) and confirmed on this season's.")]
    else:
        blocks = [note("Not run this time (--quick).")]
    r.sections.append(Section(
        "acc-halflife", "Half-life check",
        "The same deadlines replayed with half and double the half-life (every form feature and the event "
        "model's rates rebuilt), scored on the same player-gameweeks.", [card(*blocks)], tab=TAB))

    # 9 · as logged at the deadline ------------------------------------------------------------------
    logged = forecast_log.load(season.season, rounds)
    if len(logged):
        lg = logged.rename(columns={"gw": "round"})
        m = lg.merge(pg[["id", "round", "actual"]], on=["id", "round"], how="inner")
        out = []
        for g, d in m.groupby("round"):
            e = d["xp"] - d["actual"]
            out.append({"GW": int(g), "Logged at": d["logged_at"].iloc[0], "Players": len(d),
                        "Predicted": d["xp"].sum(), "Scored": d["actual"].sum(), "Bias": e.mean(),
                        "Typical miss": e.abs().mean()})
        blocks = [data_table(pd.DataFrame(out), fmt={"Predicted": "{:,.1f}", "Scored": "{:,.0f}",
                                                     "Bias": "{:+.2f}", "Typical miss": "{:.2f}"}, sortable=False)]
    else:
        blocks = [note(f"Each weekly run files its forecast for the next gameweek, injury flags and all "
                       f"(data/processed/forecast_log/). From GW{fc.next_gw}, once it's played, this section "
                       "compares the forecasts actually made at each deadline with what happened.")]
    r.sections.append(Section("acc-logged", "As forecast at the deadline", "", [card(*blocks)], tab=TAB))

    # 10 · data checks -----------------------------------------------------------------------------
    r.sections.append(Section(
        "acc-data", "Data checks",
        "That the numbers on this tab are right: the data matches FPL's official figures and the replay used "
        "nothing from after each deadline. These check the data and the method, not how good the model is.",
        [card(html_table(["Check", "Result", "Detail"],
                         [[esc(c["check"]), _status(c["ok"]), f'<div class="cell-wrap">{esc(c["detail"])}</div>']
                          for c in recon]))], tab=TAB))
