"""The HTML interface: one self-contained page you open locally, no server needed.

`build_report()` gathers the data (season store -> cross-season links -> player
profiles -> player types), and, when the weekly run passes in its results
(`WeekResults`), adds this week's tabs: This week, My squad, Fixtures, Scout, The
model (the prediction engine stage by stage) and The planner (the integer program
stage by stage), Accuracy (predicted vs actual over the last gameweeks) and The
planner. The analysis tab and a glossary of every term follow.

Each tab lives in its own module (tab_week.py, tab_squad.py, ...); the building
blocks (sections, tables, rendering, the page's script and styles) are in html.py.
`write()` produces a single HTML file with Plotly embedded inline, so it works
offline and can be archived or shared as-is. The page follows the viewer's
light/dark setting and has a toggle.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

import config
from fpl_optimiser.app import (charts, glossary, tab_accuracy, tab_decisions, tab_fixtures, tab_model,
                               tab_planner, tab_scout, tab_squad, tab_week)
from fpl_optimiser.app.common import Context, WeekResults
from fpl_optimiser.app.html import (Report, Section, card, data_table, esc, html_table, link, note,  # noqa: F401
                                    render, table)
from fpl_optimiser.app.tab_week import team_sheet  # noqa: F401  (re-exported)
from fpl_optimiser.features import opponent, player_profiles, prior_season
from fpl_optimiser.features.player_profiles import LABELS
from fpl_optimiser.ingest import opponent_stats, player_link, season_store

from fpl_optimiser.app.html import OUTPUT  # noqa: E402,F401  (where reports go)

_esc = esc


def glossary_section() -> Section:
    """Every term the page uses, grouped, each with its own #g-<slug> anchor, plus a
    filter box (the page's script hides the terms that don't match)."""
    parts = ['<input type="search" id="gloss-filter" placeholder="Filter terms, e.g. hit or xG" '
             'aria-label="Filter the glossary" autocomplete="off">']
    for title, terms in glossary.groups():
        items = "".join(f'<div class="term" id="g-{_esc(glossary.slug(term))}"><dt>{_esc(term)}</dt>'
                        f"<dd>{_esc(text)}</dd></div>" for term, text in terms)
        parts.append(f'<div class="gloss-group"><h3>{_esc(title)}</h3><dl class="gloss">{items}</dl></div>')
    parts.append('<p class="note" id="gloss-none" hidden>No terms match.</p>')
    return Section("glossary", "Glossary",
                   "What the terms on this page mean. Numbers quoted here are the settings this "
                   "report was built with (config.py).", [card("".join(parts))], tab="glossary")


# --------------------------------------------------------------------------- #
# Data -> report
# --------------------------------------------------------------------------- #
def _season_change(prev_players: pd.DataFrame, curr_players: pd.DataFrame,
                   links: pd.DataFrame, min_prev: int = config.PROFILE_MIN_MINUTES,
                   min_curr: int = config.PREV_SEASON_MIN_MINUTES) -> pd.DataFrame:
    """Points per 90 last season vs this season, for linked players who played enough."""
    def per90(p: pd.DataFrame) -> pd.DataFrame:
        mins = pd.to_numeric(p["minutes"], errors="coerce")
        return p.assign(pts_90=pd.to_numeric(p["total_points"], errors="coerce") / mins * 90,
                        mins=mins)[["id", "pts_90", "mins"]]
    m = (links[["id_prev", "id_curr"]]
         .merge(per90(prev_players).add_suffix("_prev"), on="id_prev")
         .merge(per90(curr_players).add_suffix("_curr"), on="id_curr")
         .merge(curr_players[["id", "web_name", "team_name", "position"]],
                left_on="id_curr", right_on="id"))
    m = m[(m["mins_prev"] >= min_prev) & (m["mins_curr"] >= min_curr)]
    return m.assign(delta=m["pts_90_curr"] - m["pts_90_prev"]).reset_index(drop=True)


def _add_week(r: Report, w: WeekResults, ctx: Context | None = None) -> None:
    """This week's tabs, ahead of the analysis. Each tab skips what its inputs don't
    support, so a minimal WeekResults (no team, no season context) still renders."""
    tab_week.add(r, w, ctx)
    tab_squad.add(r, w, ctx)
    tab_fixtures.add(r, w, ctx)
    tab_scout.add(r, w, ctx)
    tab_model.add(r, w, ctx)
    tab_accuracy.add(r, w, ctx)
    tab_decisions.add(r, w, ctx)
    tab_planner.add(r, w, ctx)


def build_report(season: str | None = None, week: WeekResults | None = None) -> Report:
    seasons = season_store.available()
    if not seasons:
        raise FileNotFoundError("nothing filed yet: run scripts/pull_fpl.py first")
    season = season or seasons[-1]
    cur = season_store.load(season)
    prev_season = season_store.previous_season(season)
    prev = season_store.load(prev_season) if prev_season in seasons else None
    linked = player_link.link_players(prev.players, cur.players) if prev else None
    links = linked.links if linked else None

    profiles = player_profiles.build_profiles(
        cur.players, cur.finished_gameweeks, prev.players if prev else None, links,
        prev.finished_gameweeks if prev else config.GAMEWEEKS_IN_SEASON)
    players, types = player_profiles.assign_archetypes(profiles)

    ctx = None
    if week is not None:                       # what multi_season.season_inputs builds, reusing cur/prev
        prior = (prior_season.prior_table(prev.players, links, prev_gameweeks=prev.finished_gameweeks)
                 if prev else None)
        opponents = opponent.assign_archetypes(
            opponent.build_opponent_profiles(opponent_stats.opponent_stats(season), cur.teams))
        ctx = Context(season=cur, prior=prior, profiles=opponents, types=players)

    gw = cur.finished_gameweeks
    nice = lambda s: s.replace("-", "/")                       # noqa: E731
    r = Report(title=f"FPL optimiser · {nice(season)}",
               subtitle=(f"Gameweek {week.forecast.next_gw} plan · " if week else "")
                        + f"after gameweek {gw}: predictions, player types, form and value"
                        + (f" · linked to {nice(prev_season)}" if prev else ""))
    if week is not None:
        _add_week(r, week, ctx)

    # --- overview tiles ------------------------------------------------------
    r.tiles.append(("Gameweeks played", str(gw), nice(season)))
    r.tiles.append(("Players this season", f"{len(cur.players):,}",
                    f"{len(players)} profiled ({config.PROFILE_MIN_MINUTES}+ min)"))
    if linked:
        s = linked.summary()
        check = player_link.check_against_history_past(links, cur.past, prev.players, prev_season)
        r.tiles.append(("Linked from last season", f"{s['linked']:,}",
                        f"{s['linked'] / len(cur.players):.0%} of this season's players"))
        r.tiles.append(("New arrivals", f"{s['arrivals']:,}", "no 2025/26 FPL record"))
        r.tiles.append(("Link check", f"{check['code_agrees']}/{check['checked']}",
                        "agree with FPL's own history"))

    # --- player types ----------------------------------------------------------
    sec = Section("types", "Player types",
                  "Players are grouped, position by position, by what they do per 90 minutes and how "
                  "much they play. Profiles blend this season with last season at "
                  f"{config.PREV_SEASON_WEIGHT:g}x weight per minute, so early-season noise is damped. "
                  "High and Low are relative to other profiled players in the same position. Types "
                  "are listed by points per 90, highest first.")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        if not (players["position"] == pos).any():
            continue
        xcol, ycol = charts.MAP_AXES[pos]
        cols = ["web_name", "team_name", "price", "archetype_name", "pts_90", xcol, ycol, "minutes"]
        tbl = (players[players["position"] == pos].sort_values(["archetype", "pts_90"],
                                                               ascending=[True, False])[cols]
               .rename(columns={"web_name": "Player", "team_name": "Club", "price": "Price £m",
                                "archetype_name": "Type", "pts_90": "Points/90", xcol: LABELS[xcol],
                                ycol: LABELS[ycol], "minutes": "Minutes (this season)"}))
        sec.blocks.append(card(r.chart(f"map-{pos}", charts.type_map, players, pos),
                               r.chart(f"fp-{pos}", charts.type_fingerprint, types, pos), pair=True))
        sec.blocks.append(table(tbl, f"Table: all {len(tbl)} profiled {charts.POSITION_NAMES[pos].lower()}"))
    if sec.blocks:
        r.sections.append(sec)

    # --- attacking threat ------------------------------------------------------
    att = players[players["position"].isin(["MID", "FWD"])]
    if len(att):
        top = (att.nlargest(25, "xgi_90")[["web_name", "team_name", "position", "price", "xg_90",
                                            "xa_90", "xgi_90", "pts_90"]]
               .rename(columns={"web_name": "Player", "team_name": "Club", "position": "Pos",
                                "price": "Price £m", "xg_90": "xG/90", "xa_90": "xA/90",
                                "xgi_90": "xGI/90", "pts_90": "Points/90"}))
        r.sections.append(Section("attack", "Attacking threat",
                                  "Expected goals and assists per 90 for profiled midfielders and "
                                  "forwards (blended across both seasons, as above).",
                                  [card(r.chart("attack", charts.attack_scatter, players)),
                                   table(top, "Table: top 25 by xGI/90")]))

    # --- risers and fallers -----------------------------------------------------
    change = _season_change(prev.players, cur.players, links) if prev else pd.DataFrame()
    if len(change):
        moves = pd.concat([change.nlargest(10, "delta"), change.nsmallest(10, "delta")])
        moves = moves[["web_name", "team_name", "position", "pts_90_prev", "pts_90_curr", "delta"]].rename(
            columns={"web_name": "Player", "team_name": "Club", "position": "Pos",
                     "pts_90_prev": f"Points/90 {nice(prev_season)}",
                     "pts_90_curr": f"Points/90 {nice(season)}", "delta": "Change"})
        r.sections.append(Section(
            "change", "Risers and fallers",
            f"The same players in both seasons (linked by FPL's permanent player code): points per 90 in "
            f"{nice(prev_season)} against {nice(season)} so far. Above the line = scoring more. Needs "
            f"{config.PROFILE_MIN_MINUTES}+ minutes last season and {config.PREV_SEASON_MIN_MINUTES}+ this "
            f"season ({len(change)} players).",
            [card(r.chart("change", charts.risers_fallers, change, nice(prev_season), nice(season))),
             table(moves, "Table: 10 biggest risers and 10 biggest fallers")]))

    # --- value -------------------------------------------------------------------
    regulars = profiles[profiles["minutes"] >= config.PREV_SEASON_MIN_MINUTES]
    if len(regulars):
        r.sections.append(Section("value", "Value",
                                  f"This season's points per £m of current price, for players with "
                                  f"{config.PREV_SEASON_MIN_MINUTES}+ minutes.",
                                  [card(r.chart("value", charts.value_bars, regulars))]))

    # --- the season link ---------------------------------------------------------
    if linked:
        clubs = player_link.link_teams(prev.teams, cur.teams)
        moved = clubs[clubs["status"] != "stayed"][["status", "name_prev", "name_curr"]].rename(
            columns={"status": "Status", "name_prev": nice(prev_season), "name_curr": nice(season)})
        methods = pd.DataFrame(sorted(s["by_method"].items()), columns=["Linked by", "Players"])
        arrivals = linked.arrivals["team_name"].value_counts().head(12)
        r.sections.append(Section(
            "link", f"How {nice(season)} links to {nice(prev_season)}",
            "FPL gives every player a new id each season, so seasons are joined on FPL's permanent "
            "player code, then (for anyone left) on name + date of birth, then on date of birth + "
            "nationality + surname. Only one-to-one matches are accepted.",
            [card(r.chart("arrivals", charts.count_bars, arrivals, "New arrivals by club", "Players")),
             card(f'<div class="tbl-wrap">{methods.to_html(index=False, border=0, classes="tbl")}</div>'
                  f'<div class="tbl-wrap">{moved.to_html(index=False, border=0, classes="tbl", na_rep="–")}</div>'
                  + note(f"{s['departures']} players from {nice(prev_season)} are not in "
                         f"{nice(season)}'s game; {s['changed_team']} linked players changed club and "
                         f"{s['changed_position']} changed FPL position; {s['rejected']} candidate "
                         "matches were rejected as ambiguous or contradictory."), pair=True)]))

    # --- this week's plan lives in the weekly run ---------------------------------------
    if week is None:
        r.sections.append(Section(
            "next", "This week's plan",
            "This page has the analysis only. For expected points, your transfer plan and "
            "Claude's news check, run: python scripts/run_week.py --open"))

    r.sections.append(glossary_section())

    r.footer = (f"Generated {datetime.now():%d %b %Y %H:%M} from data/raw/fpl/{season}"
                + (f" and {prev_season}" if prev else "")
                + (". Rebuild: python scripts/run_week.py" if week else ". Rebuild: python scripts/make_report.py"))
    return r
