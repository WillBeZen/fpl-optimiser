"""The HTML interface: one self-contained page you open locally, no server needed.

`build_report()` gathers the data (season store -> cross-season links -> player
profiles -> player types), draws every chart in BOTH light and dark (charts.py),
and `write()` produces a single HTML file with Plotly embedded inline, so it works
offline and can be archived or shared as-is. The page follows the viewer's
light/dark setting and has a toggle.

Later sections (predictions, squad optimiser, transfers, the Claude advisor) slot in
as they are built: each is one more `Section` added to the `Report`.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs

import config
from fpl_optimiser.app import charts
from fpl_optimiser.features import player_profiles
from fpl_optimiser.features.player_profiles import LABELS
from fpl_optimiser.ingest import player_link, season_store

OUTPUT = config.ROOT / "reports" / "fpl_report.html"
_esc = html.escape


@dataclass
class Section:
    id: str
    title: str
    intro: str = ""                                   # plain text (escaped on render)
    blocks: list[str] = field(default_factory=list)   # trusted HTML fragments, in order


@dataclass
class Report:
    title: str
    subtitle: str
    tiles: list[tuple[str, str, str]] = field(default_factory=list)   # (label, value, note)
    sections: list[Section] = field(default_factory=list)
    figures: dict[str, dict] = field(default_factory=dict)          # id -> {theme: figure}
    footer: str = ""

    def chart(self, fig_id: str, build, *args, **kwargs) -> str:
        """Draw `build(*args, theme=...)` in every theme; return its placeholder.

        The placeholder's DOM id is 'fig-<fig_id>', so it can never collide with a
        section id (getElementById returns the first match, so a clash draws the
        chart into the wrong element)."""
        self.figures[fig_id] = {theme: json.loads(pio.to_json(build(*args, theme=theme, **kwargs)))
                                for theme in charts.THEMES}
        return f'<div class="chart" id="fig-{_esc(fig_id)}"></div>'

    def write(self, path: Path = OUTPUT) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(self), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# HTML fragments
# --------------------------------------------------------------------------- #
def table(df: pd.DataFrame, caption: str, open_: bool = False) -> str:
    """A chart's table-view twin, collapsed under it (every value without hovering)."""
    body = df.to_html(index=False, border=0, classes="tbl", na_rep="–",
                      float_format=lambda v: f"{v:,.2f}")       # escapes cell text
    return (f'<details{" open" if open_ else ""}><summary>{_esc(caption)}</summary>'
            f'<div class="tbl-wrap">{body}</div></details>')


def card(*blocks: str, pair: bool = False) -> str:
    inner = "".join(f"<div>{b}</div>" for b in blocks) if pair else "".join(blocks)
    return f'<div class="card{" pair" if pair else ""}">{inner}</div>'


def note(text: str) -> str:
    return f'<p class="note">{_esc(text)}</p>'


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


def build_report(season: str | None = None) -> Report:
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

    gw = cur.finished_gameweeks
    nice = lambda s: s.replace("-", "/")                       # noqa: E731
    r = Report(title=f"FPL optimiser · {nice(season)}",
               subtitle=f"After gameweek {gw} · player types, form and value"
                        + (f" · linked to {nice(prev_season)}" if prev else ""))

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
                  f'<div class="tbl-wrap">{moved.to_html(index=False, border=0, classes="tbl")}</div>'
                  + note(f"{s['departures']} players from {nice(prev_season)} are not in "
                         f"{nice(season)}'s game; {s['changed_team']} linked players changed club and "
                         f"{s['changed_position']} changed FPL position; {s['rejected']} candidate "
                         "matches were rejected as ambiguous or contradictory."), pair=True)]))

    # --- what comes next ----------------------------------------------------------
    r.sections.append(Section(
        "next", "Coming next",
        "Expected points per player and gameweek (Section 3), the best squad, XI and captain "
        "(Section 4), when to transfer, bank or take a hit (Section 5), and Claude's read of the "
        "news with a 0-100 'how strongly to act' score (Section 6) will appear here as they are built."))

    r.footer = (f"Generated {datetime.now():%d %b %Y %H:%M} from data/raw/fpl/{season}"
                + (f" and {prev_season}" if prev else "") + ". Rebuild: python scripts/make_report.py")
    return r


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
        --muted:#898781; --line:#e1e0d9; --border:rgba(11,11,11,.10); --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
        --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#2c2c2a;
        --border:rgba(255,255,255,.10); --accent:#3987e5; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec;
        --ink2:#c3c2b7; --line:#2c2c2a; --border:rgba(255,255,255,.10); --accent:#3987e5; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width:1200px; margin:0 auto; padding:24px 16px 64px; }
header { display:flex; flex-wrap:wrap; gap:12px; align-items:flex-end; justify-content:space-between; }
h1 { font-size:26px; margin:0; font-weight:600; letter-spacing:-.01em; }
.sub { color:var(--ink2); margin:4px 0 0; }
button { font:inherit; font-size:13px; color:var(--ink2); background:var(--surface);
         border:1px solid var(--border); border-radius:8px; padding:6px 12px; cursor:pointer; }
button:hover { color:var(--ink); }
nav { display:flex; flex-wrap:wrap; gap:4px 16px; margin:16px 0 8px; font-size:14px; }
nav a { color:var(--ink2); text-decoration:none; } nav a:hover { color:var(--accent); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin:16px 0; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
.tile .label { color:var(--ink2); font-size:13px; }
.tile .value { font-size:30px; font-weight:600; line-height:1.2; }
.tile .note { color:var(--muted); font-size:12px; margin:0; }
section { margin-top:40px; }
h2 { font-size:20px; font-weight:600; margin:0 0 4px; }
.intro { color:var(--ink2); margin:0 0 12px; max-width:860px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:8px; margin:12px 0; min-width:0; }
.pair { display:grid; grid-template-columns:minmax(0, 3fr) minmax(0, 2fr); gap:12px; }
.pair > div { min-width:0; }
@media (max-width:900px) { .pair { grid-template-columns:minmax(0, 1fr); } }
.chart { width:100%; min-height:280px; }
details { margin:0 4px 16px; }
summary { cursor:pointer; color:var(--ink2); font-size:14px; padding:4px 0; }
.tbl-wrap { overflow-x:auto; }
table.tbl { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
.tbl th, .tbl td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
.tbl th { color:var(--ink2); font-weight:600; }
p.note { color:var(--ink2); font-size:13px; margin:8px 10px; }
footer { color:var(--muted); font-size:12px; margin-top:48px; }
"""

_JS = """
(function () {
  var FIGS = JSON.parse(document.getElementById('fig-data').textContent);
  var media = window.matchMedia('(prefers-color-scheme: dark)');
  var btn = document.getElementById('theme-toggle');
  var NEXT = {auto: 'light', light: 'dark', dark: 'auto'};
  var choice = 'auto';
  try { choice = localStorage.getItem('fpl-theme') || 'auto'; } catch (e) {}
  if (!NEXT[choice]) { choice = 'auto'; }
  function draw() {
    var mode = choice === 'auto' ? (media.matches ? 'dark' : 'light') : choice;
    if (choice === 'auto') { document.documentElement.removeAttribute('data-theme'); }
    else { document.documentElement.setAttribute('data-theme', choice); }
    btn.textContent = 'Theme: ' + choice;
    Object.keys(FIGS).forEach(function (id) {
      var el = document.getElementById('fig-' + id);
      if (el) { Plotly.react(el, FIGS[id][mode].data, FIGS[id][mode].layout,
                             {responsive: true, displaylogo: false}); }
    });
  }
  btn.addEventListener('click', function () {
    choice = NEXT[choice];
    try { localStorage.setItem('fpl-theme', choice); } catch (e) {}
    draw();
  });
  media.addEventListener('change', function () { if (choice === 'auto') { draw(); } });
  draw();
})();
"""


def render(r: Report) -> str:
    tiles = "".join(f'<div class="tile"><div class="label">{_esc(a)}</div><div class="value">{_esc(b)}</div>'
                    f'<p class="note">{_esc(c)}</p></div>' for a, b, c in r.tiles)
    nav = "".join(f'<a href="#{_esc(s.id)}">{_esc(s.title)}</a>' for s in r.sections)
    sections = "".join(
        f'<section id="{_esc(s.id)}"><h2>{_esc(s.title)}</h2>'
        + (f'<p class="intro">{_esc(s.intro)}</p>' if s.intro else "") + "".join(s.blocks) + "</section>"
        for s in r.sections)
    # allow_nan=False: a stray NaN would make the page's JSON.parse fail, so fail here instead
    figures = json.dumps(r.figures, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(r.title)}</title>
<style>{_CSS}</style>
<script>{get_plotlyjs()}</script>
</head>
<body>
<div class="wrap">
<header><div><h1>{_esc(r.title)}</h1><p class="sub">{_esc(r.subtitle)}</p></div>
<button id="theme-toggle" type="button">Theme: auto</button></header>
<nav>{nav}</nav>
<div class="tiles">{tiles}</div>
{sections}
<footer>{_esc(r.footer)}</footer>
</div>
<script type="application/json" id="fig-data">{figures}</script>
<script>{_JS}</script>
</body>
</html>
"""
