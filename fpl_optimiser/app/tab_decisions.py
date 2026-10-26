"""Tab · Decision test — do the model's predictions pick better teams?

Point predictions are one thing; the goal is a team that scores. This tab scores
the decisions each method would have made at past deadlines (optimise/decisions.py),
using only what was known then:

  summary   a legal team every week from each method's predictions, scored on real
            points with FPL's rules (substitutes, vice-captain); is the blend better?
  weekly    the same, gameweek by gameweek, against the hindsight-best team
  season    one squad carried through with the planner's transfers and hits, per
            method, next to keeping the first squad and to your real team
  captains  how each method's captains did
  last season  the weekly test and the season with transfers on last season, from
            the study that chose the settings (model/tuning.py): many more gameweeks
"""
from __future__ import annotations

import pandas as pd

import config
from fpl_optimiser.app import charts
from fpl_optimiser.app.common import Context, WeekResults, pct, player_info
from fpl_optimiser.app.html import Report, Section, card, data_table, kv, note
from fpl_optimiser.model import evaluation
from fpl_optimiser.optimise import decisions

TAB = "decisions"
LABELS = {**evaluation.METHODS, decisions.HINDSIGHT: "Hindsight best (upper bound)", "you": "Your actual team"}


def _verdict(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return "Too few gameweeks"
    return "Blend better" if low > 0 else ("This method better" if high < 0 else "No clear difference")


def _weekly_tables(picks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(each method's weekly summary, the table of it, the blend against each method)."""
    ws = decisions.weekly_summary(picks)
    order = [m for m in decisions.METHODS if m in set(ws["method"])] + [decisions.HINDSIGHT]
    ws = ws.set_index("method").loc[order].reset_index()
    wtab = pd.DataFrame({"Method": ws["method"].map(LABELS), "Points a gameweek": ws["points"],
                         "Spread (SD)": ws["points_sd"], "Captain's points a gameweek": ws["captain_points"],
                         "Captain was the XI's best": ws["captain_best"], "Share of hindsight": ws["of_hindsight"]})
    pw = decisions.paired_weeks(picks, "points")
    pw = pw[pw["method"] != decisions.HINDSIGHT]
    ptab = pd.DataFrame({"Method": pw["method"].map(LABELS), "Blend minus method (points a GW)": pw["mean_diff"],
                         "90% interval": [f"{lo:+.1f} to {hi:+.1f}" if pd.notna(lo) else "–"
                                          for lo, hi in zip(pw["low"], pw["high"])],
                         "Blend better in": [f"{b} of {g} GWs" for b, g in zip(pw["reference_better"], pw["gameweeks"])],
                         "Verdict": [_verdict(lo, hi) for lo, hi in zip(pw["low"], pw["high"])]})
    return ws, wtab, ptab


WEEKLY_FMT = {"Points a gameweek": "{:.1f}", "Spread (SD)": "{:.1f}", "Captain's points a gameweek": "{:.1f}",
              "Captain was the XI's best": pct, "Share of hindsight": pct}


def _season_table(ss: pd.DataFrame) -> pd.DataFrame:
    ss = ss.set_index("method")
    ss = ss.loc[[m for m in decisions.METHODS if m in ss.index]].reset_index()
    return pd.DataFrame({"Method": ss["method"].map(LABELS), "Points after hits": ss["net"], "Points": ss["points"],
                         "Hit cost": ss["hit_cost"], "Transfers": ss["transfers"], "Hits": ss["hits"],
                         "Captain's points": ss["captain_points"], "Kept the first squad": ss["held"],
                         "Gain from transfers": ss["transfer_gain"]})


SEASON_FMT = ({c: "{:.0f}" for c in ["Points after hits", "Points", "Hit cost", "Captain's points", "Kept the first squad"]}
              | {"Gain from transfers": "{:+.0f}"})


def _last_season(r: Report, w: WeekResults) -> None:
    """The decision test on last season (from the study that chose the settings)."""
    st = w.last_season
    d = (st or {}).get("decisions")
    if not d:
        return
    season, picks = st["season"], d["picks"]
    rounds = sorted(picks["round"].unique())
    ws, wtab, ptab = _weekly_tables(picks)
    blend = ws.set_index("method").loc[evaluation.REFERENCE]
    r.sections.append(Section(
        "dec-last-season", f"Last season ({season}): {len(rounds)} gameweeks",
        f"The same two tests on last season, replayed deadline by deadline (half-life {st['chosen']['half_life']:g}, "
        f"as chosen there). With {len(rounds)} gameweeks the paired comparisons are much firmer than this season's.",
        [card(kv([("Gameweeks", f"{len(rounds)} (GW{rounds[0]}-GW{rounds[-1]})"),
                  ("The blend's team", f"{blend['points']:.1f} pts a GW"),
                  ("Hindsight best", f"{ws.set_index('method').at[decisions.HINDSIGHT, 'points']:.1f} pts a GW")]),
              "<h3>A legal team every week</h3>", data_table(wtab, fmt=WEEKLY_FMT, sortable=False),
              "<h3>Is the blend's team better than each method's?</h3>",
              data_table(ptab, fmt={"Blend minus method (points a GW)": "{:+.1f}"}, sortable=False),
              f"<h3>A season with transfers, GW{rounds[0]}-GW{rounds[-1]}</h3>",
              data_table(_season_table(d["season"]), fmt=SEASON_FMT, sortable=False),
              note("The season with transfers is one path through the season (early choices compound), so its "
                   "ranking is illustrative; the weekly test is the paired comparison."))], tab=TAB))


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    d = w.decisions
    if not d or w.evaluation is None:
        r.sections.append(Section("dec-summary", "Do the predictions pick better teams?", "",
                                  [note("Not run this time (--quick).")], tab=TAB))
        _last_season(r, w)
        return
    ev = w.evaluation
    picks, seasons = d["picks"], d["seasons"]
    rounds = ev.rounds
    span = f"GW{rounds[0]}-GW{rounds[-1]}"
    ref = evaluation.REFERENCE

    # 1 · summary: a free hit every week -------------------------------------------------
    ws, wtab, ptab = _weekly_tables(picks)
    order = list(ws["method"])
    blend = ws.set_index("method").loc[ref]
    best_simple = ws[~ws["method"].isin([ref, "xp", "xp_events", decisions.HINDSIGHT])].nlargest(1, "points").iloc[0]
    hind = ws.set_index("method").loc[decisions.HINDSIGHT]
    r.sections.append(Section(
        "dec-summary", f"Do the predictions pick better teams? {span}",
        "Each gameweek, every method picks the best legal team it can see from its own predictions as they "
        "stood at the deadline: 15 players within £100m at that week's prices, at most 3 per club, a legal XI, a "
        "captain and a vice. The team is then scored on FPL's real points with FPL's rules (substitutes come on "
        "for starters who didn't play; the vice-captain's points double if the captain didn't play). This is a "
        "free hit every week: it tests picking, without the constraint of last week's squad.",
        [card(kv([("Gameweeks", f"{len(rounds)} ({span})"),
                  ("The blend's team", f"{blend['points']:.1f} pts a GW"),
                  (f"Best simple method ({LABELS[best_simple['method']]})", f"{best_simple['points']:.1f} pts a GW"),
                  ("Hindsight best", f"{hind['points']:.1f} pts a GW")]),
              data_table(wtab, fmt=WEEKLY_FMT, sortable=False),
              "<h3>Is the blend's team better than each method's?</h3>",
              data_table(ptab, fmt={"Blend minus method (points a GW)": "{:+.1f}"}, sortable=False),
              note("Captain's points: what the armband added (the captain's points once more). The hindsight team is "
                   "picked knowing every real score, so it is a ceiling, not a target. Paired by gameweek, with a 90% "
                   "interval: with this few gameweeks, only large, consistent differences show."))], tab=TAB))

    # 2 · gameweek by gameweek ------------------------------------------------------------
    shown = [m for m in [ref, "xp", "pos_price_p", decisions.HINDSIGHT] if m in set(picks["method"])]
    wide = picks.pivot(index="round", columns="method", values="points")
    wide = wide[[m for m in order if m in wide.columns]].rename(columns=LABELS).reset_index().rename(columns={"round": "GW"})
    r.sections.append(Section(
        "dec-weekly", "Gameweek by gameweek",
        "Each method's team score every gameweek, with the hindsight-best team above them all.",
        [card(r.chart("dec-weekly", charts.method_lines, picks.rename(columns={"points": "value"}), "value", shown,
                      LABELS, "Real points of each method's team (a free hit every week)", "Points"),
              data_table(wide, fmt={c: "{:.0f}" for c in LABELS.values()}, sortable=False))], tab=TAB))

    # 3 · a season with transfers -----------------------------------------------------------
    stab = _season_table(decisions.season_summary(seasons))
    per_week = pd.concat([s.weeks.assign(method=m) for m, s in seasons.items()])
    cum = per_week.sort_values("round").assign(value=lambda d: d.groupby("method")["net"].cumsum())
    lines = [m for m in [ref, "pos_price_p", "xp"] if m in seasons]
    actual = d.get("actual")
    blocks = []
    if actual is not None and len(actual):
        mine = actual[actual["round"].isin(rounds)].sort_values("round")
        if len(mine):
            net = (mine["points"] - mine["hit_cost"])
            you = pd.DataFrame({"method": "you", "round": mine["round"], "value": net.cumsum()})
            cum = pd.concat([cum, you], ignore_index=True)
            lines.append("you")
            chips_used = ", ".join(f"{c.replace('_', ' ')} GW{g}" for g, c in zip(mine["round"], mine["chip"])
                                   if isinstance(c, str))
            blocks.append(note(f"Your actual team over {span}: {net.sum():.0f} points after {mine['hit_cost'].sum():.0f} "
                               f"in hits" + (f" (chips: {chips_used})" if chips_used else "") + ". Not a like-for-like "
                               "comparison: your squad, chips and timing differ, and the simulated squads start from "
                               f"scratch at GW{rounds[0]} with no chips."))
    net_tab = per_week.pivot(index="round", columns="method", values="net")
    net_tab = net_tab[[m for m in decisions.METHODS if m in net_tab.columns]].rename(columns=LABELS)
    net_tab = net_tab.reset_index().rename(columns={"round": "GW"})
    r.sections.append(Section(
        "dec-season", "A season with transfers",
        f"One squad per method, picked from scratch at GW{rounds[0]} and then managed by the planner from that "
        f"method's multi-week predictions: free transfers banked or spent, a -{config.TRANSFER_HIT_COST} hit only "
        f"when it clears {config.HIT_THRESHOLD_POINTS:g} points, FPL's selling prices, no chips. 'Kept the first "
        "squad' plays the same first 15 every week (best XI and captain each time); the difference is what the "
        "transfers gained after hits. One path through one stretch of fixtures: early choices compound, so treat "
        "the ranking as illustrative.",
        [card(data_table(stab, fmt=SEASON_FMT, sortable=False),
              r.chart("dec-season", charts.method_lines, cum, "value", lines, LABELS,
                      "Running total of points after hits", "Points"),
              *blocks,
              "<h3>Points after hits, gameweek by gameweek</h3>",
              data_table(net_tab, fmt={c: "{:.0f}" for c in LABELS.values()}, sortable=False))], tab=TAB))

    # 4 · captains -------------------------------------------------------------------------
    info = player_info(w, ctx) if ctx is not None else None
    refp = picks[picks["method"] == ref].sort_values("round")
    cap = pd.DataFrame({"GW": refp["round"],
                        "The blend's captain": refp["captain_id"].map(info["web_name"]) if info is not None else refp["captain_id"],
                        "Captain's points (doubled)": refp["captain_points"] * config.CAPTAIN_MULTIPLIER,
                        "Best in that XI": refp["best_captain_points"] * config.CAPTAIN_MULTIPLIER})
    r.sections.append(Section(
        "dec-captains", "Captains",
        "Which player each week's blend team captained, and what the best captain in that same XI would have "
        "scored. The captain is the starter with the most expected points; often someone else in the XI "
        "scores more, which is the nature of picking one player out of eleven.",
        [card(data_table(cap, fmt={"Captain's points (doubled)": "{:.0f}", "Best in that XI": "{:.0f}"}, sortable=False))],
        tab=TAB))

    _last_season(r, w)
