"""Tab 6 · The planner — the integer program, stage by stage, with your numbers.

  1 inputs           your squad (bought for / now / sells for), bank, free transfers, dials
  2 candidate pool   which players are worth giving to the solver
  3 the problem      decision variables, rules and the objective, and their sizes
  4 baseline         keep everyone: what your squad scores as it is
  5 how many moves   the best plan with 0, 1, 2, ... transfers this week
  6 the plan         week by week: transfers, hits, chip, bank, free transfers carried
  7 chips            chips left per set, when the plan plays them, and what each chip
                     would add in each gameweek against its threshold
  8 why these moves  each move's gain and the next-best alternatives
  9 what's binding   club limits, money, formation; what more money would buy
 10 the objective    the number the solver maximised, piece by piece
 11 captain, bench   each week's armband and bench order
 12 a wildcard       the best squad from scratch, for comparison
"""
from __future__ import annotations

import pandas as pd

import config
from fpl_optimiser.app import charts
from fpl_optimiser.app.common import Context, WeekResults, money, player_info
from fpl_optimiser.app.html import Report, Section, card, data_table, kv, note, stage, table
from fpl_optimiser.optimise import planner, pool, whatif

TAB = "planner"


def _sheet(team: pd.DataFrame, fixtures: pd.Series | None = None) -> pd.DataFrame:
    from fpl_optimiser.app.tab_week import team_sheet
    return team_sheet(team, fixtures)


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    fc = w.forecast
    nxt_fx = fc.xp[fc.xp["gw"] == fc.next_gw].set_index("id")["fixtures"]
    if w.plan is None or w.baseline is None or w.team is None:
        _wildcard(r, w, nxt_fx, 1)
        return
    plan, base, t = w.plan, w.baseline, w.team
    info = player_info(w, ctx)
    first = plan.weeks[0]

    # 1 · inputs -----------------------------------------------------------------------
    rows = []
    for pid in t.squad_ids:
        p = info.loc[pid] if pid in info.index else None
        bought = t.purchase_prices.get(pid)
        now = p["price"] if p is not None else None
        sells = t.sell_prices.get(pid, now)
        rows.append({"Pos": p["position"] if p is not None else "", "Player": p["web_name"] if p is not None else pid,
                     "Club": p["team_name"] if p is not None else "", "Bought for": bought, "Now": now,
                     "Sells for": sells, "Lost to the sell-on rule": (now - sells) if now is not None and sells is not None else None})
    sq = pd.DataFrame(rows).sort_values("Pos", key=lambda s: s.map({p: n for n, p in enumerate(config.POSITION_QUOTA)}))
    dials = pd.DataFrame([
        ("Horizon", f"{len(fc.gameweeks)} gameweeks (GW{fc.gameweeks[0]}-{fc.gameweeks[-1]})", "PLANNING_HORIZON_GW"),
        ("Horizon decay", f"each week counts {config.HORIZON_DECAY:g}x the one before", "HORIZON_DECAY"),
        ("Hit threshold", f"a -{config.TRANSFER_HIT_COST} hit must gain > {config.HIT_THRESHOLD_POINTS:g} pts",
         "HIT_THRESHOLD_POINTS"),
        ("Banked transfer value", f"{config.FREE_TRANSFER_VALUE:g} pts each at the end", "FREE_TRANSFER_VALUE"),
        ("Bench weight", f"bench points count {config.BENCH_WEIGHT:g}x", "BENCH_WEIGHT"),
        ("Chip thresholds", ", ".join(f"{planner.CHIP_NAMES[k]} {v:g}" for k, v in config.CHIP_THRESHOLD_POINTS.items())
         + " pts (in that week's points)", "CHIP_THRESHOLD_POINTS"),
        ("Free-transfer cap", f"at most {config.MAX_BANKED_TRANSFERS} banked", "MAX_BANKED_TRANSFERS"),
        ("Club limit", f"at most {config.MAX_PER_CLUB} per club", "MAX_PER_CLUB"),
    ], columns=["Setting", "Value", "config.py"])
    r.sections.append(Section("p-inputs", "1 · Inputs", "", [
        stage(1, "Inputs", "What the planner starts from: your 15 at FPL's selling prices (you keep half of "
              "any rise, rounded down to £0.1m), your bank and free transfers, every player's expected points "
              "for each gameweek of the horizon, and the strategy dials."),
        card(kv([("Bank", money(t.bank)), ("Free transfers", str(t.free_transfers)),
                 ("Squad value at selling prices", money(sum(t.sell_prices.get(i, 0) for i in t.squad_ids))),
                 ("Picks from", f"GW{t.gw}")]),
             data_table(sq, fmt={"Bought for": money, "Now": money, "Sells for": money,
                                 "Lost to the sell-on rule": money})),
        table(dials, "Table: the strategy dials", open_=True)], tab=TAB))

    # 2 · candidate pool -----------------------------------------------------------------
    kept = pool.prune(fc.xp, fc.players, keep=t.squad_ids)
    by_pos = []
    for pos, quota in config.POSITION_QUOTA.items():
        ids = set(fc.players.loc[fc.players["position"] == pos, "id"])
        by_pos.append({"Position": pos, "Players": len(ids), "Kept": len(ids & kept),
                       "Dropped as dominated": len(ids - kept), "Squad needs": quota})
    s = plan.stats
    r.sections.append(Section("p-pool", "2 · Pool", "", [
        stage(2, "The candidate pool", f"Most players can never be in the best squad: at least "
              f"quota + {pool.SLACK} others at their position cost no more and score at least as much in "
              f"every gameweek. Dropping them leaves {s.get('candidates', '?')} of {s.get('players', '?')} "
              "players and the same answer, several times faster. Your own players are always kept."),
        card(data_table(pd.DataFrame(by_pos), sortable=False))], tab=TAB))

    # 3 · the problem --------------------------------------------------------------------
    T = len(fc.gameweeks)
    r.sections.append(Section("p-problem", "3 · Problem", "", [
        stage(3, "The integer program", "Every decision is a yes/no variable; every rule a linear "
              "constraint. The solver (CBC) finds the plan with the highest objective that breaks no rule: "
              "the true optimum for these expected points, not a rule of thumb."),
        card(kv([("Decision variables", f"{s.get('variables', 0):,}"), ("of which yes/no", f"{s.get('binary', 0):,}"),
                 ("Constraints", f"{s.get('constraints', 0):,}"), ("Chip options", str(s.get("chip_options", 0))),
                 ("Built in", f"{s.get('build_seconds', 0):.1f}s"), ("Solved in", f"{s.get('solve_seconds', 0):.1f}s"),
                 ("Proven optimal", "yes" if s.get("proven_optimal", True) else
                  f"no (stopped after {planner.CHIP_SOLVE_SECONDS}s)")]),
             '<div class="formula">'
             f"for each player i and each of the {T} gameweeks t:\n"
             "  squad[i,t]  start[i,t]  captain[i,t]  vice[i,t]  buy[i,t]  sell[i,t]   (yes/no)\n"
             f"for each gameweek:  free[t] (0-{config.MAX_BANKED_TRANSFERS})  hits[t]  bank[t] (£m, >= 0)\n"
             "for each chip still available, each gameweek it's allowed:  chip[k,t]   (yes/no)\n\n"
             "rules, every gameweek:\n"
             f"  {config.SQUAD_SIZE} in the squad: " + ", ".join(f"{n} {p}" for p, n in config.POSITION_QUOTA.items())
             + f"; {config.XI_SIZE} start in a legal formation; 1 captain, 1 vice, both starting\n"
             f"  at most {config.MAX_PER_CLUB} per club\n"
             "  squad[t] = squad[t-1] + bought - sold            bank[t] = bank[t-1] + sales - purchases\n"
             "  hits[t] >= transfers[t] - free[t]                 free[t+1] <= free[t] - transfers + hits + 1\n"
             f"  at most one chip a week; each chip once per set (GW1-{config.CHIP_SPLIT_GW}, "
             f"GW{config.CHIP_SPLIT_GW + 1}-{config.GAMEWEEKS_IN_SEASON})\n"
             "  wildcard: that week's transfers cost no hits     free hit: a one-week squad, yours rests\n"
             f"  bench boost: the bench scores in full            triple captain: captain x{config.TRIPLE_CAPTAIN_MULTIPLIER}\n\n"
             "maximise:\n"
             f"  sum over t of {config.HORIZON_DECAY:g}^t x (XI points + captain's points again + "
             f"{config.BENCH_WEIGHT:g} x bench points)\n"
             f"  - {config.HIT_THRESHOLD_POINTS:g} x hits - each chip's threshold (decayed) "
             f"+ {config.FREE_TRANSFER_VALUE:g} x free transfers left at the end"
             "</div>")], tab=TAB))

    # 4 · baseline -----------------------------------------------------------------------
    bl = pd.DataFrame({"GW": [x.gw for x in base.weeks], "Captain": [x.captain for x in base.weeks],
                       "Formation": [x.formation for x in base.weeks], "Expected pts": [x.xp for x in base.weeks],
                       "With the plan": [x.xp for x in plan.weeks]})
    bl["Plan gains"] = bl["With the plan"] - bl["Expected pts"]
    r.sections.append(Section("p-baseline", "4 · Baseline", "", [
        stage(4, "The baseline: no transfers", f"Your squad kept as it is for all {T} gameweeks, with "
              "its best XI and captain each week. Every plan is measured against this: "
              f"{base.xp:.1f} expected points over the horizon, against {plan.net:.1f} with the plan after hits."),
        card(data_table(bl, fmt={"Expected pts": "{:.1f}", "With the plan": "{:.1f}", "Plan gains": "{:+.1f}"}))],
        tab=TAB))

    # 5 · how many moves -------------------------------------------------------------------
    blocks = [stage(5, "How many transfers this week?",
                    "The same problem re-solved with exactly 0, 1, 2, ... transfers this gameweek (later "
                    "weeks free to do their best, any chip the plan plays kept where it is). Each bar is what "
                    "one more transfer adds; once a move costs a hit the bar must clear "
                    f"{config.HIT_THRESHOLD_POINTS:g} to be worth it.")]
    if w.ladder is not None and len(w.ladder):
        lad = w.ladder
        blocks.append(card(r.chart("p-ladder", charts.ladder_bars, lad, len(first.bought)),
                           data_table(lad[["transfers", "hits", "out", "in", "xp_next", "net", "objective", "gain"]].rename(
                               columns={"transfers": "Transfers", "hits": "Hits", "out": "Out", "in": "In",
                                        "xp_next": f"GW{fc.next_gw} pts", "net": "Horizon pts after hits",
                                        "objective": "Objective", "gain": "Gain"}),
                               fmt={f"GW{fc.next_gw} pts": "{:.1f}", "Horizon pts after hits": "{:.1f}",
                                    "Objective": "{:.2f}", "Gain": "{:+.2f}"}, sortable=False)))
    else:
        blocks.append(note("Not run this time (--quick)."))
    r.sections.append(Section("p-ladder", "5 · How many", "", blocks, tab=TAB))

    # 6 · the plan -------------------------------------------------------------------------
    rows = [{"GW": x.gw, "Free transfers": x.free_transfers, "Out": ", ".join(x.sold["web_name"]) or "-",
             "In": ", ".join(x.bought["web_name"]) or "-", "Hits": x.hits,
             "Chip": planner.CHIP_NAMES[x.chip] if x.chip else "-", "Carried to next week": x.free_after,
             "Bank after": x.bank_after, "Formation": x.formation, "Captain": x.captain, "Expected pts": x.xp}
            for x in plan.weeks]
    r.sections.append(Section("p-plan", "6 · The plan", "", [
        stage(6, "The plan, week by week", "Only this gameweek's moves are acted on; later weeks show "
              "what the plan expects to do next, and it is re-planned every week as new data arrives. "
              "On a free hit, Out and In are that week's one-off squad changes; your squad is back "
              "the week after."),
        card(data_table(pd.DataFrame(rows), fmt={"Bank after": money, "Expected pts": "{:.1f}"}, sortable=False))],
        tab=TAB))

    # 7 · chips ---------------------------------------------------------------------------
    r.sections.append(_chips(w, plan, t))

    # 8 · why these moves --------------------------------------------------------------------
    blocks = [stage(8, "Why these transfers", "Each move against the player it replaces, then the "
                    "next-best players who could have come in instead (same position, affordable with that "
                    "slot's money, club limit kept), ranked by the decayed horizon points the planner weighs.")]
    if len(first.bought):
        why = whatif.transfer_reasons(first, fc.xp)
        gw_cols = [c for c in why.columns if c.startswith("GW")]
        blocks.append(card(data_table(why, fmt={"Sells for £m": "{:.1f}", "Costs £m": "{:.1f}",
                                                "Money freed £m": "{:+.1f}", "Gain over horizon": "{:+.2f}",
                                                **{c: "{:+.2f}" for c in gw_cols}})))
        alt = whatif.alternatives(first, fc.xp, fc.players)
        alt = alt.rename(columns={"slot": "Slot", "budget": "Money for it", "player": "Player", "club": "Club",
                                  "price": "Price", "xp_horizon": "Horizon pts", "weighted": "Decayed pts",
                                  "vs_chosen": "vs chosen", "chosen": "Chosen"})
        alt["Chosen"] = alt["Chosen"].map({True: "✓", False: ""})
        blocks.append(table(alt.round(2), "Table: the alternatives for each move", open_=True))
        blocks.append(note("An alternative can rate slightly higher on its own and still lose: the plan "
                           "weighs the whole squad together (who starts, who sits on the bench at "
                           f"{config.BENCH_WEIGHT:g}x, and later weeks' moves)."))
    else:
        blocks.append(note("No transfers this week."))
    r.sections.append(Section("p-why", "8 · Why", "", blocks, tab=TAB))

    # 9 · what's binding -----------------------------------------------------------------------
    use = whatif.constraint_use(first)
    clubs = use["clubs"].rename(columns={"club": "Club", "players": "Players", "at_limit": "At the limit"})
    clubs["At the limit"] = clubs["At the limit"].map({True: "yes", False: ""})
    form = pd.DataFrame([{"Position": p, "Starting": n, "Allowed": f"{lo}-{hi}"} for p, (n, lo, hi) in use["formation"].items()])
    blocks = [stage(9, "What's binding", f"After this week's moves: £{use['bank_after']:.1f}m left in the bank, "
                    f"{use['free_after']} free transfer(s) carried. Clubs at the {config.MAX_PER_CLUB}-player limit "
                    "block further buys from them."),
              card(data_table(clubs, sortable=False), data_table(form, sortable=False), pair=True)]
    if w.budget is not None and len(w.budget) > 1:
        bud = w.budget.rename(columns={"extra": "Extra £m", "net": "Horizon pts after hits", "objective": "Objective",
                                       "gain": "Gain", "in": "Would buy this week"})
        blocks.append(card(r.chart("p-budget", charts.budget_bars, w.budget),
                           data_table(bud, fmt={"Extra £m": "+£{:g}m", "Horizon pts after hits": "{:.1f}",
                                                "Objective": "{:.2f}", "Gain": "{:+.2f}"}, sortable=False)))
    r.sections.append(Section("p-limits", "9 · Limits", "", blocks, tab=TAB))

    # 10 · the objective -----------------------------------------------------------------------
    parts = whatif.objective_parts(plan).rename(columns={"part": "Part", "weight": "Decay weight", "xi": "XI pts",
                                                         "captain": "Captain again", "bench": "Bench pts",
                                                         "value": "Counts for"})
    r.sections.append(Section("p-objective", "10 · Objective", "", [
        stage(10, "The objective, piece by piece", f"What the solver maximised: {plan.objective:.2f}. Each "
              "gameweek's XI, captain and bench (as its chip scores them), weighted by the horizon decay, "
              "minus the hit penalty and the thresholds of the chips played, plus the value of transfers "
              "still banked at the end. (Tiny tie-breaks make up any rounding gap.)"),
        card(data_table(parts, fmt={"Decay weight": "{:.2f}", "XI pts": "{:.1f}", "Captain again": "{:.1f}",
                                    "Bench pts": "{:.1f}", "Counts for": "{:.2f}"}, sortable=False))], tab=TAB))

    # 11 · captain and bench -----------------------------------------------------------------
    rows = []
    for x in plan.weeks:
        tm = x.team
        bench = tm[tm["role"] == "bench"].sort_values("bench_order") if "bench_order" in tm else tm[tm["role"] == "bench"]
        vice = tm.loc[tm["vice"], "web_name"]
        rows.append({"GW": x.gw, "Chip": planner.CHIP_NAMES[x.chip] if x.chip else "-",
                     "Captain": x.captain, "Vice": vice.iloc[0] if len(vice) else "-",
                     "Captain's pts": float(tm.loc[tm["captain"], "xp"].sum()),
                     "Bench (in order)": ", ".join(bench["web_name"]),
                     "Bench pts": float(bench["xp"].sum())})
    r.sections.append(Section("p-captain", "11 · Captain & bench", "", [
        stage(11, "Captain and bench each week", "The captain is the starter with the most expected points "
              f"(his points count twice, {config.TRIPLE_CAPTAIN_MULTIPLIER} times on a triple captain); the "
              "vice is the next best. The bench is ordered for automatic substitutions: reserve keeper "
              "first, then outfielders by expected points. On a bench boost all 15 score."),
        card(data_table(pd.DataFrame(rows), fmt={"Captain's pts": "{:.2f}", "Bench pts": "{:.1f}"}, sortable=False)),
        table(_sheet(first.team, nxt_fx), f"Table: your GW{fc.next_gw} team sheet", open_=True)], tab=TAB))

    _wildcard(r, w, nxt_fx, 12)


def _chips(w: WeekResults, plan, t) -> Section:
    """Stage 7: chips left, the plan's chip decisions, and what each chip would add."""
    fc, names = w.forecast, planner.CHIP_NAMES
    last = fc.gameweeks[-1]
    played = ", ".join(f"the {names[x.chip]} in GW{x.gw}" for x in plan.weeks if x.chip)
    blocks = [stage(7, "Chips", "The plan may play a chip in any gameweek of the horizon: at most one a "
                    f"week, each once per set (the first set by the GW{config.CHIP_SPLIT_GW} deadline, the "
                    "second after it). A chip is played only if it gains more than its threshold in that "
                    "week's points (" + ", ".join(f"{names[k]} {v:g}" for k, v in config.CHIP_THRESHOLD_POINTS.items())
                    + "): roughly what it should be worth saved for a double or blank gameweek. A set that "
                    "expires within the horizon has no such value, so its chips go in their best remaining "
                    "week. " + (f"This plan plays {played}." if played else
                                f"This plan plays no chip in GW{fc.gameweeks[0]}-{last}."))]
    if t.chips_available:
        left = pd.DataFrame([{"Set": f"GW{a}-{b}", "Chips left": ", ".join(names[k] for k in chips) or "none",
                              "In the horizon": _in_horizon(a, b, fc.gameweeks[0], last)}
                             for (a, b), chips in t.chips_available.items()])
        blocks.append(card(data_table(left, sortable=False)))
    else:
        blocks.append(note("No chips available to plan with."))
    if not plan.stats.get("proven_optimal", True):
        blocks.append(note(f"The full chip problem stopped after {planner.CHIP_SOLVE_SECONDS}s, so this is the "
                           "best plan found rather than a proven optimum. It is never worse than its starting "
                           "plan: the best plan without chips, with the chips worth playing on its squads."))
    grid = w.chips
    if grid is not None and len(grid):
        best = grid.loc[grid.groupby("chip")["gain"].idxmax()].set_index("chip").reindex(
            [k for k in config.CHIPS if k in set(grid["chip"])])
        rows = []
        for chip, b in best.iterrows():
            if chip in plan.chips:
                verdict = f"planned for GW{plan.chips[chip]}"
            elif b["worth"]:
                verdict = "worth it, but the plan uses that week (or this chip) better"
            else:
                verdict = f"not yet: {b['must_beat'] - b['gain']:.1f} short"
            rows.append({"Chip": names[chip], "Set": b["set"], "Best week": f"GW{b['gw']}",
                         "Adds (expected pts)": b["xp_gain"], "Planner's gain": b["gain"],
                         "Must beat": b["must_beat"], "Verdict": verdict})
        blocks.append(card("<h3>Each chip in its best week</h3>",
                           note("Each chip played alone in its best gameweek, everything else re-planned "
                                "around it, against the plan with no chips: the expected points it adds over "
                                "the horizon (after hits), and its gain in the planner's decayed points next to "
                                "the threshold it must beat that week."),
                           data_table(pd.DataFrame(rows), fmt={"Adds (expected pts)": "{:+.1f}",
                                                               "Planner's gain": "{:+.2f}", "Must beat": "{:.2f}"},
                                      sortable=False)))
        gws = sorted(grid["gw"].unique())
        hi = max(float(grid["gain"].max()), 1.0)
        cells = []
        for chip in [k for k in config.CHIPS if k in set(grid["chip"])]:
            row = {"Chip": names[chip]}
            for gw in gws:
                g = grid[(grid["chip"] == chip) & (grid["gw"] == gw)]
                if g.empty:
                    row[f"GW{gw}"] = "–"
                    continue
                g = g.iloc[0]
                mark = " ✓" if g["planned"] else ""
                row[f"GW{gw}"] = (float(g["gain"]), f"{g['gain']:+.1f}<small> / {g['must_beat']:.1f}</small>{mark}")
            cells.append(row)
        cols = [f"GW{gw}" for gw in gws]
        blocks.append(card("<h3>Chip in each gameweek</h3>",
                           note("Each cell: the planner's gain from playing that chip that week (alone, the rest "
                                "re-planned), then the threshold it must beat that week. ✓ = in the plan. Shaded by "
                                "gain; – = not allowed that week."),
                           data_table(pd.DataFrame(cells), heat={c: (0, hi) for c in cols}, html_cols=tuple(cols),
                                      sortable=False)))
    else:
        blocks.append(note("What each chip would add in each gameweek: not run this time (--quick)."))
    return Section("p-chips", "7 · Chips", "", blocks, tab=TAB)


def _in_horizon(first: int, last: int, h_first: int, h_last: int) -> str:
    """How a chip set (GW first-last) meets the horizon (GW h_first-h_last)."""
    if first > h_last:
        return "not yet"
    if last <= h_last:
        return f"expires after GW{last}: use it or lose it"
    return f"from GW{first}" if first > h_first else "yes"


def _wildcard(r: Report, w: WeekResults, nxt_fx: pd.Series, n: int) -> None:
    s = w.wildcard
    r.sections.append(Section("wildcard", f"{n} · Wildcard", "", [
        stage(n, "If you had a wildcard", f"The best squad from scratch for GW{w.forecast.next_gw} on "
              f"expected points: {s.formation}, £{s.cost:.1f}m, {s.xp:.1f} expected points with {s.captain} "
              "as captain. A useful yardstick: the gap to your plan is what your current squad costs you."),
        table(_sheet(s.players, nxt_fx), f"Table: the GW{w.forecast.next_gw} squad", open_=True)], tab=TAB))
