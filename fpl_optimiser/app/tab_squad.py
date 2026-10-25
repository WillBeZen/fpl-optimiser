"""Tab 2 · My squad — what each of your players should do in each fixture.

Your squad today plus this week's buys: expected points per gameweek with the
fixture, then the next gameweek event by event (minutes, xG, xA, chance of a goal,
an assist, a clean sheet, a defensive-contribution bonus), where the points come
from, how each player rates against every club, and a player explorer with any
player's full picture (form story, fixtures, set pieces).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from fpl_optimiser.app import charts
from fpl_optimiser.app.common import (Context, WeekResults, has_events, money, own, pct, player_info,
                                      squad_roles)
from fpl_optimiser.app.html import Report, Section, bars, card, data_table, esc, kv, note, pill, table
from fpl_optimiser.model import events, explain

TAB = "squad"
POS_ORDER = {p: n for n, p in enumerate(config.POSITION_QUOTA)}
EVENT_FMT = {"xMins": "{:.0f}", "P(plays)": pct, "P(60+)": pct, "xG": "{:.2f}", "xA": "{:.2f}",
             "P(goal)": pct, "P(assist)": pct, "P(clean sheet)": pct, "P(def. contr.)": pct,
             "Saves": "{:.1f}", "Bonus": "{:.2f}", "Event model": "{:.2f}", "Points model": "{:.2f}",
             "Expected pts": "{:.2f}"}


def _who(w: WeekResults, ctx: Context | None) -> tuple[pd.DataFrame, str]:
    """The players this tab covers, with Keep/Sell/Buy (or the from-scratch squad)."""
    roles = squad_roles(w)
    if len(roles):
        return roles, "your squad today, plus this week's buys"
    return pd.DataFrame({"id": list(w.wildcard.players["id"]), "plan": "Pick"}), \
        "the best squad from scratch (add FPL_TEAM_ID to .env to see your own)"


def event_rows(fx: pd.DataFrame) -> pd.DataFrame:
    """Per-fixture rows -> the event columns a scout reads."""
    out = pd.DataFrame({
        "Fixture": fx["fixture"], "xMins": fx["xmins"], "P(plays)": fx["p_play"], "P(60+)": fx["p_60"],
        "xG": fx["xg"], "xA": fx["xa"], "P(goal)": fx["p_goal"], "P(assist)": fx["p_assist"],
        "P(clean sheet)": fx["p_cs"], "P(def. contr.)": fx["p_defcon"], "Saves": fx["saves"],
        "Bonus": fx["Bonus"], "Event model": fx["xp_events"], "Points model": fx["xp_model"],
        "Expected pts": fx["xp"]}, index=fx.index)
    return out


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    fc = w.forecast
    roles, whose = _who(w, ctx)
    info = player_info(w, ctx)
    ids = [i for i in roles["id"] if i in info.index]
    plan_of = roles.set_index("id")["plan"]
    sell = (w.team.sell_prices if w.team is not None else {})
    if not ids:
        if has_events(w):
            explorer(r, w, ctx, [])
        return

    # --- the grid: expected points per gameweek, with the fixture ----------------------
    grid = fc.xp[fc.xp["id"].isin(ids)]
    hi = max(float(fc.xp.groupby("gw")["xp"].quantile(0.99).max()), 1.0)
    rows = []
    for pid in sorted(ids, key=lambda i: (plan_of[i] == "Sell", POS_ORDER.get(info.at[i, "position"], 9),
                                          -grid[grid["id"] == i]["xp"].sum())):
        p = info.loc[pid]
        row = {"Plan": plan_of[pid], "Pos": p["position"], "Player": p["web_name"], "Club": p["team_short"]
               if isinstance(p["team_short"], str) else p["team_name"]}
        total = 0.0
        for g in fc.gameweeks:
            cell = grid[(grid["id"] == pid) & (grid["gw"] == g)]
            v = float(cell["xp"].sum()) if len(cell) else 0.0
            total += v
            fx_txt = cell["fixtures"].iloc[0] if len(cell) else "-"
            row[f"GW{g}"] = (v, f"{v:.1f}<small>{esc(fx_txt)}</small>")
        row["Total"] = total
        row.update({"Price": p["price"],
                    "Sells for": sell.get(pid, p["price"]) if plan_of[pid] != "Buy" else float("nan"),
                    "Owned": p["selected_by_percent"],
                    "Status": (p["news"] if isinstance(p["news"], str) and p["news"] else "Available")})
        rows.append(row)
    df = pd.DataFrame(rows)
    gw_cols = [f"GW{g}" for g in fc.gameweeks]
    r.sections.append(Section(
        "squad-grid", "Expected points, fixture by fixture",
        f"For {whose}: expected points in each of the next {len(fc.gameweeks)} gameweeks (the blend "
        "the planner uses, after FPL's injury flags), with the opponent under each number. Darker = more "
        "points. Click a column to sort.",
        [card(data_table(df, fmt={"Price": money, "Sells for": money, "Owned": own, "Total": "{:.1f}"},
                         heat={**{c: (0, hi) for c in gw_cols}}, html_cols=tuple(gw_cols),
                         row_attrs=lambda row: {"class": "mine"} if row["Plan"] in ("Keep", "Buy") else {}))],
        tab=TAB))

    if not has_events(w):
        return
    fx = fc.per_fixture[fc.per_fixture["id"].isin(ids)].copy()
    fx["Player"] = fx["id"].map(info["web_name"])
    fx["Plan"] = fx["id"].map(plan_of)

    # --- the next gameweek, event by event -------------------------------------------
    nxt = fx[fx["round"] == fc.next_gw].sort_values(["Plan", "xp"], ascending=[True, False])
    ev = pd.concat([nxt[["Plan", "Player"]], event_rows(nxt)], axis=1)
    r.sections.append(Section(
        "squad-next", f"GW{fc.next_gw}, event by event",
        "What each player is expected to do in his next match: expected minutes, xG and xA, the "
        "chances of a goal, an assist, a clean sheet (60+ minutes, team concedes nothing) and the "
        "defensive-contribution bonus, saves for keepers and bonus points. 'Event model' adds these up "
        "with FPL's scoring; 'Points model' is the trees' direct estimate; 'Expected pts' is their "
        f"blend ({1 - fc.event_weight:g} / {fc.event_weight:g}), which the planner uses.",
        [card(data_table(ev, fmt=EVENT_FMT, heat={"Expected pts": (0, 8)}))], tab=TAB))

    # --- where the points come from ----------------------------------------------------------
    src = nxt.groupby("id")[events.SOURCES + ["xp_events"]].sum()
    src["label"] = [f"{info.at[i, 'web_name']} · {info.at[i, 'position']}" for i in src.index]
    src = src.sort_values("xp_events", ascending=False)
    tbl = src[["label"] + events.SOURCES + ["xp_events"]].rename(columns={"label": "Player", "xp_events": "Total"})
    r.sections.append(Section(
        "squad-sources", f"Where the points come from, GW{fc.next_gw}",
        "The event model's expected points split by FPL scoring source. Appearance points are the "
        "floor every starter gets; attackers earn through goals and assists, defenders through "
        "clean sheets and defensive contributions (minus goals conceded).",
        [card(r.chart("squad-sources", charts.sources_bars, src.reset_index()),
              table(tbl.round(2), "Table: every source"))], tab=TAB))

    # --- the whole horizon, fixture by fixture ----------------------------------------------
    hz = fx.sort_values(["Player", "round"])
    hz_tbl = pd.concat([hz[["Player"]], hz[["round"]].rename(columns={"round": "GW"}), event_rows(hz)], axis=1)
    r.sections.append(Section(
        "squad-horizon", "Every fixture in the horizon", "",
        [data_table(hz_tbl, fmt=EVENT_FMT, heat={"Expected pts": (0, 8)}, tall=True,
                    caption=f"Table: {len(hz_tbl)} player-fixtures, GW{fc.gameweeks[0]}-{fc.gameweeks[-1]}",
                    open_=False)], tab=TAB))

    # --- against every club ------------------------------------------------------------------
    m = (explain.opponent_matrix(fc, ids, ctx.profiles, ctx.season.fixtures, ctx.season.teams)
         if ctx is not None and ctx.season is not None and ctx.profiles is not None else pd.DataFrame())
    if len(m):
        m.index = [f"{info.at[i, 'web_name']} ({info.at[i, 'team_short'] if isinstance(info.at[i, 'team_short'], str) else ''})"
                   for i in m.index]
        order = m.mean().sort_values(ascending=False).index
        m = m[order]
        r.sections.append(Section(
            "squad-opponents", "Against every club",
            "Each player's expected points against each opponent, averaged over home and away, "
            "assuming he's fit: his own form and minutes, with the opponent's defence (and for "
            "defenders, attack) swapped in. Clubs run from the easiest opponents on the left to "
            "the hardest on the right.",
            [card(r.chart("squad-opponents", charts.heatmap, m, "Expected points vs each club",
                          colorbar="Exp. pts")),
             table(m.round(2).reset_index(names="Player"), "Table: every value")], tab=TAB))

    explorer(r, w, ctx, ids)


def explorer(r: Report, w: WeekResults, ctx: Context | None, first_ids: list[int], extra: int = 40) -> None:
    """A panel per player (your squad first, then the top expected scorers), picked from
    a dropdown: who he is, his set-piece duties, his fixtures event by event, and his
    form story with each match's recency weight."""
    fc = w.forecast
    info = player_info(w, ctx)
    totals = fc.xp.groupby("id")["xp"].sum().sort_values(ascending=False)
    ids = list(dict.fromkeys(list(first_ids) + [i for i in totals.index[:extra]]))
    ids = [i for i in ids if i in info.index]
    history = ctx.season.history if ctx is not None and ctx.season is not None else None
    teams = ctx.season.teams if ctx is not None and ctx.season is not None else None
    prior = ctx.prior if ctx is not None else None
    hl = explain.half_life_table(history, ids, fc.next_gw, prior) if history is not None else None

    options, panels = [], []
    for k, pid in enumerate(ids):
        p = info.loc[pid]
        mine = pid in set(first_ids)
        options.append(f'<option value="{pid}"{" selected" if k == 0 else ""}>'
                       f'{esc(p["web_name"])} · {esc(p["team_name"])} {esc(p["position"])}'
                       f'{" (yours)" if mine and w.team is not None else ""}</option>')
        duties = [lbl for col, lbl in (("penalties_order", "penalties"),
                                       ("corners_and_indirect_freekicks_order", "corners"),
                                       ("direct_freekicks_order", "free kicks"))
                  if p.get(col) == 1]
        head = kv([("Club", p["team_name"]), ("Position", p["position"]), ("Price", money(p["price"])),
                   ("Owned by", own(p["selected_by_percent"])),
                   (f"Next {len(fc.gameweeks)} GWs", f"{totals.get(pid, 0):.1f} pts"),
                   ("Set pieces", ", ".join(duties) or "none first-choice"),
                   ("FPL's own estimate, next GW", f"{p['ep_next']:.1f}" if pd.notna(p["ep_next"]) else "–")])
        status = (pill(p["news"], "warn") if isinstance(p["news"], str) and p["news"] else pill("Available", "good"))
        parts = [f'<h3>{esc(p["web_name"])}</h3>', f'<p>{status}</p>', head]
        if has_events(w):
            pf = fc.per_fixture[fc.per_fixture["id"] == pid].sort_values("round")
            if len(pf):
                t = pd.concat([pf[["round"]].rename(columns={"round": "GW"}), event_rows(pf)], axis=1)
                parts.append("<h3>Fixtures, event by event</h3>" + data_table(t, fmt=EVENT_FMT, sortable=False))
        if history is not None:
            m, s = explain.form_story(history, pid, fc.next_gw, prior, teams)
            if len(m):
                parts.append("<h3>Form story: each match and how much it counts</h3>")
                parts.append(bars([(f"GW{rd}", float(pts), f"GW{rd} {opp}: {pts:g} pts, {mins:g} min, "
                                    f"weight {wt:.2f} ({sh:.0%} of form)")
                                   for rd, opp, pts, mins, wt, sh in
                                   zip(m["round"], m["opponent"], m["points"], m["minutes"], m["weight"], m["share"])],
                                  label=f"{p['web_name']}'s points per match",
                                  sub=[(float(sh / max(m["share"].max(), 1e-9)), f"{sh:.0%}") for sh in m["share"]]))
                txt = (f"Form (what the model sees): {s['form']:.2f} points a game; this season's plain "
                       f"average {s['plain_average']:.2f} over {s['matches']} matches")
                if not np.isnan(s["last_season"]):
                    txt += (f"; last season {s['last_season']:.2f} a gameweek, still "
                            f"{s['last_season_share']:.0%} of his form")
                parts.append(note(txt + ". The small bars under each match are its share of the form."))
                if hl is not None and pid in hl.index:
                    parts.append(note("With a different half-life his form would be: " + ", ".join(
                        f"{c} {v:.2f}" for c, v in hl.loc[pid].items()) + "."))
        panels.append(f'<div class="explore" data-player="{pid}"{"" if k == 0 else " hidden"}>{card("".join(parts))}</div>')

    picker = (f'<div class="explorer-pick"><label for="explorer-select">Player</label>'
              f'<select id="explorer-select" data-explorer="explorer-panels">{"".join(options)}</select></div>')
    r.sections.append(Section(
        "explorer", "Player explorer",
        "Pick any player: your squad first, then the top expected scorers over the horizon.",
        [picker, f'<div id="explorer-panels">{"".join(panels)}</div>'], tab=TAB))
