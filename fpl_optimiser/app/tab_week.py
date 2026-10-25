"""Tab 1 · This week — what to do before the deadline, at a glance.

The plan's moves for the next gameweek and why, the plan week by week against
making no transfers, your team sheet, alerts about your players (injuries, price
changes, blanks, rotation risk), Claude's news check and the captain options.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

import config
from fpl_optimiser.app import charts
from fpl_optimiser.app.common import Context, WeekResults, next_fixtures, own, pct, player_info
from fpl_optimiser.app.html import Report, Section, card, data_table, esc, html_table, link, note, table
from fpl_optimiser.optimise import whatif

TAB = "week"


def team_sheet(team: pd.DataFrame, fixtures: pd.Series | None = None) -> pd.DataFrame:
    """The 15 as a table (XI then bench), captain (C) and vice (V) marked."""
    mark = team["captain"].map({True: " (C)"}).fillna("") + team["vice"].map({True: " (V)"}).fillna("")
    out = pd.DataFrame({"": team["role"].map({"XI": "XI", "bench": "Bench"}),
                        "Pos": team["position"], "Player": team["web_name"] + mark,
                        "Club": team["team_name"]})
    if fixtures is not None:
        out["Fixture"] = team["id"].map(fixtures).fillna("-")
    out["Price £m"] = team["price"]
    out["Expected pts"] = team["xp"].round(2)
    return out


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    fc = w.forecast
    gw, last = fc.next_gw, fc.gameweeks[-1]
    nxt_fx = fc.xp[fc.xp["gw"] == gw].set_index("id")["fixtures"]
    when = (f"{datetime.strptime(fc.deadline, '%Y-%m-%d %H:%M'):%a %d %b, %H:%M} UTC"
            if fc.deadline else "")
    tiles = [("Next deadline", f"GW{gw}", when)]

    sec = Section("week", "Your plan", tab=TAB)
    if w.plan is not None and w.baseline is not None:
        first = w.plan.weeks[0]
        gain = w.plan.net - w.baseline.xp
        tiles += [("Plan vs no transfers", f"{gain:+.1f} pts", f"GW{gw}-{last}, after hits"),
                  ("Transfers this week", str(len(first.bought)),
                   f"{first.hits} hit(s), {first.free_transfers} free available"),
                  ("Captain", first.captain, f"team expects {first.xp:.1f} pts in GW{gw}")]
        t = w.team
        sec.intro = (f"From your squad (GW{t.gw} picks), £{t.bank:.1f}m in the bank and "
                     f"{t.free_transfers} free transfer(s) (estimated from your history; override "
                     f"with --free-transfers), the planner looked {len(fc.gameweeks)} gameweeks "
                     f"ahead. A -4 hit is only taken if it gains more than "
                     f"{config.HIT_THRESHOLD_POINTS:g} expected points. {t.note}").strip()
        if len(first.bought):
            why = whatif.transfer_reasons(first, fc.xp)
            gw_cols = [c for c in why.columns if c.startswith("GW")]
            sec.blocks.append(card(
                f"<h3>This week's moves</h3>",
                note("Each move: what the player out sells for, what the player in costs, and the "
                     "expected points it gains per gameweek. A move that loses points can still be "
                     "right when the money it frees pays for a bigger upgrade elsewhere."),
                data_table(why, fmt={"Sells for £m": "{:.1f}", "Costs £m": "{:.1f}",
                                     "Money freed £m": "{:+.1f}", "Gain over horizon": "{:+.2f}",
                                     **{c: "{:+.2f}" for c in gw_cols}})))
        else:
            sec.blocks.append(note(f"No transfers this week: the free transfer is banked "
                                   f"({first.free_after} available next week)."))
        rows = []
        for pw, bw in zip(w.plan.weeks, w.baseline.weeks):
            rows.append({"GW": pw.gw,
                         "Out": ", ".join(pw.sold["web_name"]) or "-",
                         "In": ", ".join(pw.bought["web_name"]) or "-",
                         "Hits": pw.hits, "Free transfers": pw.free_transfers,
                         "Captain": pw.captain, "Expected pts": round(pw.xp, 1),
                         "No transfers": round(bw.xp, 1)})
        sec.blocks += [
            card(r.chart("plan", charts.plan_vs_baseline, [x.gw for x in w.plan.weeks],
                         [x.xp for x in w.plan.weeks], [x.xp for x in w.baseline.weeks])),
            table(pd.DataFrame(rows), "Table: the plan week by week", open_=True),
            table(team_sheet(first.team, nxt_fx), f"Table: your GW{gw} team sheet", open_=True)]
    else:
        tiles.append(("Best squad from scratch", f"{w.wildcard.xp:.1f} pts",
                      f"GW{gw}, captain {w.wildcard.captain}"))
        sec.blocks.append(note("Add FPL_TEAM_ID=<your team id> to .env (the number in your "
                               "team's URL on the FPL site) to get a transfer plan for your own squad."))
    r.sections.append(sec)

    if w.team is not None:
        _alerts(r, w, ctx)
    _advice(r, w, tiles)
    _captaincy(r, w, ctx)
    r.tiles[:0] = tiles


def _alerts(r: Report, w: WeekResults, ctx: Context | None) -> None:
    """Things about your players worth knowing before the deadline."""
    fc = w.forecast
    info = player_info(w, ctx)
    ids = [i for i in w.team.squad_ids if i in info.index]
    nxt = fc.xp[fc.xp["gw"] == fc.next_gw].set_index("id")
    rows = []
    for i in ids:
        p = info.loc[i]
        name = f"{p['web_name']} ({p['team_name']})"
        if p["status"] not in ("a", None) and isinstance(p["status"], str):
            rows.append(["Availability", esc(name), esc(p["news"] or p["status"])])
        change = p.get("price_change_percent")
        if pd.notna(change) and abs(change) >= 50:
            rows.append(["Price", esc(name), esc(f"FPL's price-change progress {change:+.0f}%: a "
                                                 f"{'rise' if change > 0 else 'fall'} may be close")])
        blanks = [g for g in fc.gameweeks if fc.xp[(fc.xp["id"] == i) & (fc.xp["gw"] == g)]["fixtures"].eq("-").any()]
        doubles = [g for g in fc.gameweeks
                   if fc.xp[(fc.xp["id"] == i) & (fc.xp["gw"] == g)]["fixtures"].str.contains(",").any()]
        if blanks:
            rows.append(["Fixtures", esc(name), esc("No match in " + ", ".join(f"GW{g}" for g in blanks))])
        if doubles:
            rows.append(["Fixtures", esc(name), esc("Plays twice in " + ", ".join(f"GW{g}" for g in doubles))])
        if i in nxt.index and p["status"] == "a" and nxt.at[i, "p_60"] < 0.5 and nxt.at[i, "p_play"] > 0:
            rows.append(["Minutes", esc(name), esc(f"Only {nxt.at[i, 'p_60']:.0%} to play 60+ minutes in "
                                                   f"GW{fc.next_gw} (rotation risk)")])
    blocks = [card(html_table(["About", "Player", "What"], rows))] if rows else [
        note("Nothing flagged: everyone is available, no price moves are close, and no blanks or "
             "doubles in the horizon.")]
    r.sections.append(Section("alerts", "Alerts for your squad",
                              "Injuries and suspensions (FPL's own news), price changes FPL says are "
                              "close, blank and double gameweeks in the horizon, and rotation risk.",
                              blocks, tab=TAB))


def _advice(r: Report, w: WeekResults, tiles: list) -> None:
    adv = Section("advice", "News check (Claude)", tab=TAB)
    a = w.advice
    if a is None:
        adv.blocks.append(note("Skipped this run (--no-advisor)."))
    elif not a.ok:
        adv.blocks.append(note(f"Advisor unavailable: {a.error}"))
    else:
        tiles.append(("News check", f"{a.action_score}/100", f"Claude says: {a.verdict}"))
        spend = (f" This check used {a.input_tokens:,} input and {a.output_tokens:,} output "
                 f"tokens, about ${a.cost:.2f} at list prices."
                 if a.cost is not None and (a.input_tokens or a.output_tokens) else "")
        when = f" Checked {a.checked_at} UTC." if a.checked_at else ""
        adv.intro = (f"{a.model} read the latest news for the players involved "
                     f"({a.searches} web searches) and scored how strongly to act on this "
                     "week's moves. News adjusts, never overrides, the maths." + when + spend)
        verdict = (f'<div class="verdict"><div class="value">{a.action_score}<span>/100</span></div>'
                   f'<div><p class="big">{esc(a.verdict.capitalize())}</p><p>{esc(a.summary)}</p>'
                   f'<p class="note">Captain: {esc(a.captain_note)}</p></div></div>')
        flags = html_table(["Player", "Status", "Note", "Source"],
                           [[esc(f["player"]), esc(f["status"]), esc(f["note"]),
                             link(f["source"], "source") if f["source"] else "-"] for f in a.player_flags])
        lists = "".join(f"<h3>{esc(title)}</h3><ul>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>"
                        for title, items in (("Suggested adjustments", a.adjustments), ("Risks", a.risks))
                        if items)
        seen, srcs = set(), []
        for s in a.sources:
            if s["url"] not in seen:
                seen.add(s["url"])
                srcs.append(f"<li>{link(s['url'], s['title'] or s['url'])}</li>")
        sources = f"<details><summary>Sources searched ({len(srcs)})</summary><ul>{''.join(srcs[:15])}</ul></details>"
        adv.blocks.append(card(verdict + flags + lists + sources))
    r.sections.append(adv)


def _captaincy(r: Report, w: WeekResults, ctx: Context | None) -> None:
    """The captain options in your XI (or the from-scratch XI) next gameweek."""
    fc = w.forecast
    team = w.plan.weeks[0].team if w.plan is not None else w.wildcard.players
    xi = team[team["role"] == "XI"].set_index("id")
    fx = next_fixtures(w)
    if fx.empty or "fixture" not in fx:
        fx = pd.DataFrame({"id": pd.Series(dtype=int), "fixture": pd.Series(dtype=str),
                           "p_play": pd.Series(dtype=float)})
    per = fx[fx["id"].isin(xi.index)].groupby("id").agg(
        fixture=("fixture", ", ".join), p_play=("p_play", "max"),
        **({"p_goal": ("p_goal", lambda s: 1 - (1 - s).prod()),
            "p_assist": ("p_assist", lambda s: 1 - (1 - s).prod()),
            "xg": ("xg", "sum")} if "p_goal" in fx else {}))
    info = player_info(w, ctx)
    rows = []
    for pid, p in xi.sort_values("xp", ascending=False).head(6).iterrows():
        f = per.loc[pid] if pid in per.index else None
        rows.append({"": "C" if p["captain"] else ("V" if p["vice"] else ""),
                     "Player": p["web_name"], "Club": p["team_name"],
                     "Fixture": f["fixture"] if f is not None else "-",
                     "Expected pts": p["xp"], "x2 as captain": 2 * p["xp"],
                     "P(plays)": f["p_play"] if f is not None else float("nan"),
                     "P(goal)": f["p_goal"] if f is not None and "p_goal" in f else float("nan"),
                     "P(assist)": f["p_assist"] if f is not None and "p_assist" in f else float("nan"),
                     "Owned by": info["selected_by_percent"].get(pid, float("nan"))})
    df = pd.DataFrame(rows)
    r.sections.append(Section(
        "captain", f"Captain options, GW{fc.next_gw}",
        "The six highest expected scorers in your XI. The captain's points count double, so the "
        "gap between the top two matters twice. Ownership matters for rank: captaining a widely "
        "owned player protects it, a less owned one is a gamble for gains.",
        [card(data_table(df, fmt={"Expected pts": "{:.2f}", "x2 as captain": "{:.2f}", "P(plays)": pct,
                                  "P(goal)": pct, "P(assist)": pct, "Owned by": own}))], tab=TAB))
