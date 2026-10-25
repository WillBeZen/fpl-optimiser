"""Tab 3 · Fixtures — which clubs to attack with and which to defend with.

Team ratings from the event model (attack = xG created per game, defence = xG
conceded per game, recency-weighted with last season as a fading prior), then a
fixture ticker over the planning horizon: each club's expected goals and its
clean-sheet chance, gameweek by gameweek, sorted so the best runs are on top.
"""
from __future__ import annotations

import pandas as pd

from fpl_optimiser.app import charts
from fpl_optimiser.app.common import Context, WeekResults, pct
from fpl_optimiser.app.html import Report, Section, card, data_table, esc, note
from fpl_optimiser.model import events, explain

TAB = "fixtures"


def ticker(outlook: pd.DataFrame, gameweeks: list[int], value: str, fmt) -> pd.DataFrame:
    """Club x gameweek cells of (sum of `value`, "OPP (H)" text), sorted by total;
    a blank gameweek is 0, a double adds both fixtures."""
    rows = []
    for team, d in outlook.groupby("team"):
        row = {"Club": team}
        total = 0.0
        for g in gameweeks:
            f = d[d["gw"] == g]
            v = float(f[value].sum()) if len(f) else 0.0
            total += v
            text = ", ".join(f["vs"]) if len(f) else "no match"
            row[f"GW{g}"] = (v, f"{fmt(v)}<small>{esc(text)}</small>")
        row["Total"] = total
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)


def add(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    fc = w.forecast
    if fc.ratings is None or ctx is None or ctx.season is None:
        return
    rt = fc.ratings
    teams = rt.teams.sort_values("att_index", ascending=False)
    tbl = pd.DataFrame({"Club": teams["name"], "Attack (xG/game)": teams["attack"],
                        "Defence (xG against/game)": teams["defence"],
                        "Attack vs average": teams["att_index"], "Defence vs average": teams["def_index"],
                        "Matches this season": teams["matches"], "Last season's share": teams["prior_share"],
                        "Promoted": teams["promoted"].map({True: "yes", False: ""})})
    r.sections.append(Section(
        "ratings", "Team ratings",
        f"Each club's attack and defence as of GW{fc.next_gw}: expected goals created and conceded per "
        f"game, recent matches counting more, with last season's goals blended in as a fading prior "
        f"(promoted clubs start from the relegated clubs' record). League average {rt.mean_goals:.2f} "
        f"goals a team a game; home sides score {rt.home:.2f}x that, away sides {rt.away:.2f}x.",
        [card(r.chart("ratings", charts.ratings_scatter, teams)),
         card(data_table(tbl, fmt={"Attack (xG/game)": "{:.2f}", "Defence (xG against/game)": "{:.2f}",
                                   "Attack vs average": "{:.2f}x", "Defence vs average": "{:.2f}x",
                                   "Last season's share": pct}))], tab=TAB))

    out = events.fixture_outlook(rt, ctx.season.fixtures, fc.gameweeks)
    gw_cols = [f"GW{g}" for g in fc.gameweeks]
    att = ticker(out, fc.gameweeks, "xg_for", lambda v: f"{v:.2f}")
    cs = ticker(out, fc.gameweeks, "p_cs", lambda v: f"{v:.0%}")
    r.sections.append(Section(
        "ticker-attack", "Fixture ticker: goals",
        "Each club's expected goals per gameweek from its fixtures (home advantage included). Pick "
        "attackers from the clubs at the top. A double gameweek adds both matches; a blank is 0.",
        [card(data_table(att, html_cols=tuple(gw_cols), fmt={"Total": "{:.2f}"},
                         heat={c: (0, 3.0) for c in gw_cols}))], tab=TAB))
    r.sections.append(Section(
        "ticker-defence", "Fixture ticker: clean sheets",
        "Each club's chance of keeping a clean sheet per gameweek (the opponent scoring 0, from its "
        "expected goals). Total = expected clean sheets over the horizon. Pick defenders and keepers "
        "from the clubs at the top.",
        [card(data_table(cs, html_cols=tuple(gw_cols), fmt={"Total": "{:.2f}"},
                         heat={c: (0, 0.6) for c in gw_cols}))], tab=TAB))

    if ctx.profiles is not None:
        names = explain.style_names(ctx.profiles)
        prof = ctx.profiles.assign(style=ctx.profiles["archetype"].map(names))
        styles = (prof.groupby("style", sort=False)
                  .agg(clubs=("team", lambda s: ", ".join(sorted(s))), gc=("gc_pg", "mean"))
                  .reset_index().rename(columns={"style": "Style", "clubs": "Clubs",
                                                 "gc": "Goals conceded a game"}))
        r.sections.append(Section(
            "styles", "Defensive styles",
            "The points model also sees each opponent by its defensive 'style': the 20 clubs "
            "clustered on goals conceded and FPL's defence ratings (k-means), tightest first. The "
            "Model tab shows what each type of player has scored against each style.",
            [card(data_table(styles, fmt={"Goals conceded a game": "{:.2f}"}, sortable=False))], tab=TAB))
    else:
        r.sections.append(Section("styles", "Defensive styles", "", [note("Not available.")], tab=TAB))
