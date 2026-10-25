"""HTML building blocks for the report: one self-contained page, no server needed.

A `Report` holds tiles, `Section`s and Plotly figures. Each section belongs to a
tab (This week, My squad, Fixtures, Scout, The model, The planner, Player analysis,
Glossary); `render()` lays the tabs out, inlines Plotly and a small script, and
returns one HTML string that works offline.

Fragments:
  card, note, link, table (a DataFrame under a "Table:" toggle), html_table
  data_table   a sortable (optionally filterable) table with heat-coloured cells
  heat_class   a value -> one of seven steps of the validated blue ramp
  stage        a numbered step heading for the model / planner walk-throughs
  bars         a small CSS bar chart (no Plotly) for compact panels
  kv           a row of label / value pairs

Untrusted text (player names, FPL news, everything Claude returns) is always
escaped; links are http(s) only.
"""
from __future__ import annotations

import html
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs

import config
from fpl_optimiser.app import charts

esc = html.escape
OUTPUT = config.ROOT / "reports" / "fpl_report.html"

TABS = [("week", "This week"), ("squad", "My squad"), ("fixtures", "Fixtures"),
        ("scout", "Scout"), ("model", "The model"), ("accuracy", "Accuracy"), ("planner", "The planner"),
        ("analysis", "Player analysis"), ("glossary", "Glossary")]
HEAT_STEPS = 7


@dataclass
class Section:
    id: str
    title: str
    intro: str = ""                                   # plain text (escaped on render)
    blocks: list[str] = field(default_factory=list)   # trusted HTML fragments, in order
    tab: str = "analysis"


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
        return f'<div class="chart" id="fig-{esc(fig_id)}"></div>'

    def write(self, path: Path | None = None) -> Path:
        path = path or OUTPUT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(self), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# Fragments
# --------------------------------------------------------------------------- #
def table(df: pd.DataFrame, caption: str, open_: bool = False) -> str:
    """A chart's table-view twin, collapsed under it (every value without hovering)."""
    body = df.to_html(index=False, border=0, classes="tbl", na_rep="–",
                      float_format=lambda v: f"{v:,.2f}")       # escapes cell text
    return (f'<details{" open" if open_ else ""}><summary>{esc(caption)}</summary>'
            f'<div class="tbl-wrap">{body}</div></details>')


def card(*blocks: str, pair: bool = False) -> str:
    inner = "".join(f"<div>{b}</div>" for b in blocks) if pair else "".join(blocks)
    return f'<div class="card{" pair" if pair else ""}">{inner}</div>'


def note(text: str) -> str:
    return f'<p class="note">{esc(text)}</p>'


def link(url, text: str) -> str:
    """An outbound link from untrusted data: http(s) only, escaped, opened in a new
    tab without passing a referrer. Anything else is shown as plain text."""
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return esc(text)
    return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(text)}</a>'


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    """A table whose cells are already-escaped HTML (e.g. containing link())."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def heat_class(value, lo: float, hi: float) -> str:
    """'h0'..'h6': where `value` sits between lo and hi (blank for missing)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or hi <= lo:
        return ""
    step = int(np.clip((float(value) - lo) / (hi - lo) * HEAT_STEPS, 0, HEAT_STEPS - 1))
    return f"h{step}"


# Over / under-prediction (predicted - actual), on the palette's diverging pair: blue =
# scored MORE than predicted, red = scored LESS, grey = about right. Steps (points):
ERROR_EDGES = (-4.5, -2.5, -0.75, 0.75, 2.5, 4.5)


def error_class(err) -> str:
    """'e0' (scored far more than predicted) .. 'e3' (about right) .. 'e6' (far less)."""
    if err is None or (isinstance(err, float) and np.isnan(err)):
        return ""
    return f"e{int(np.searchsorted(ERROR_EDGES, float(err), side='right'))}"


def _fmt(v, f) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return "–"
    if callable(f):
        return f(v)
    if f:
        return f.format(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}"
    return str(v)


def data_table(df: pd.DataFrame, *, id: str | None = None, fmt: dict | None = None,
               heat: dict | None = None, html_cols: tuple = (), sortable: bool = True,
               row_attrs: Callable[[pd.Series], dict] | None = None, tall: bool = False,
               caption: str | None = None, open_: bool = True, hidden: tuple = ()) -> str:
    """A table from a DataFrame: every cell escaped (except `html_cols`), numbers
    formatted by `fmt` ({column: "{:.1f}" or a function}), and `heat` columns
    ({column: (lo, hi)}) shaded on the blue ramp. Sortable by clicking a header
    (numeric cells carry their raw value in data-v). A cell may also be a
    (number, trusted html) pair: sorted and shaded by the number, shown as the html.
    `row_attrs(row)` adds data-* attributes (used by the filters); a "class" key sets
    the row's class; `hidden` columns feed row_attrs but aren't shown. With
    `caption`, the table sits under a toggle."""
    fmt, heat = fmt or {}, heat or {}
    numeric = (int, float, np.integer, np.floating)
    shown = [c for c in df.columns if c not in hidden]
    head = "".join(f'<th scope="col">{esc(str(c))}</th>' for c in shown)
    rows = []
    for r in df.to_dict("records"):            # not iterrows: it turns ints into floats in mixed rows
        attrs = ""
        if row_attrs:
            for k, v in row_attrs(r).items():
                attrs += f' class="{esc(str(v))}"' if k == "class" else f' data-{esc(k)}="{esc(str(v))}"'
        cells = []
        for c in shown:
            v = r[c]
            own = ""
            if isinstance(v, tuple):
                num, text = v[0], v[1]
                own = v[2] if len(v) > 2 else ""
            else:
                num = v
                text = str(v) if c in html_cols else esc(_fmt(v, fmt.get(c)))
            ok = isinstance(num, numeric) and not isinstance(num, bool) and not np.isnan(float(num))
            dv = f' data-v="{float(num):.6g}"' if ok else ""
            cls = own or (heat_class(float(num), *heat[c]) if ok and c in heat else "")
            cells.append(f"<td{dv}" + (f' class="{cls}"' if cls else "") + f">{text}</td>")
        rows.append(f"<tr{attrs}>{''.join(cells)}</tr>")
    cls = "tbl" + (" sortable" if sortable else "")
    ident = f' id="{esc(id)}"' if id else ""
    out = (f'<div class="tbl-wrap{" tall" if tall else ""}"><table class="{cls}"{ident}>'
           f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    if caption:
        return f'<details{" open" if open_ else ""}><summary>{esc(caption)}</summary>{out}</details>'
    return out


def stage(n: int | str, title: str, text: str = "") -> str:
    """A numbered step in a walk-through, used as a section's first block: it becomes
    the section's heading ('Stage 3' over 'Team characteristics') in place of the
    title, which then only labels the section in the tab's menu."""
    return (f'<h2 class="stage"><span class="stage-n">Stage {esc(str(n))}</span>'
            f'<span class="stage-t">{esc(title)}</span></h2>' + (f'<p class="intro">{esc(text)}</p>' if text else ""))


def kv(pairs: list[tuple[str, str]]) -> str:
    """Label / value pairs in a row (values escaped)."""
    items = "".join(f'<div><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>' for k, v in pairs)
    return f'<div class="kv">{items}</div>'


def bars(items: list[tuple[str, float, str]], max_value: float | None = None,
         label: str = "", sub: list[tuple[float, str]] | None = None) -> str:
    """A compact CSS bar chart: items are (axis label, value, tooltip). `sub` adds a
    second, smaller bar row under each bar (value 0..1, text), e.g. a weight."""
    top = max_value or max([abs(v) for _, v, _ in items] + [1e-9])
    cols = []
    for k, (lbl, v, tip) in enumerate(items):
        h = 0 if top == 0 else max(0.0, v) / top * 100
        extra = ""
        if sub:
            sv, st = sub[k]
            extra = f'<div class="sub" style="--s:{sv * 100:.0f}%"><i></i><b>{esc(st)}</b></div>'
        cols.append(f'<div class="col" title="{esc(tip)}"><div class="bar-area"><div class="bar" style="--h:{h:.1f}%">'
                    f'<span>{esc(f"{v:g}")}</span></div></div>{extra}<div class="lbl">{esc(lbl)}</div></div>')
    return f'<div class="bars" role="img" aria-label="{esc(label)}">{"".join(cols)}</div>'


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {esc(kind)}">{esc(text)}</span>'


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_HEAT_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_HEAT_LIGHT_INK = ["#0b0b0b"] * 4 + ["#ffffff"] * 3
_HEAT_DARK = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#3987e5", "#6da7ec", "#9ec5f4"]
_HEAT_DARK_INK = ["#f0efec"] * 4 + ["#0b0b0b"] * 3
# diverging blue <-> grey <-> red, each step's ink chosen for >= 4.5:1 contrast
_ERR_LIGHT = ["#256abf", "#729dd0", "#b7cadf", "#f0efec", "#e4bdba", "#d57f7e", "#c43b3a"]
_ERR_LIGHT_INK = ["#ffffff"] + ["#0b0b0b"] * 5 + ["#ffffff"]
_ERR_DARK = ["#6da7ec", "#597da6", "#475768", "#383835", "#694543", "#a45554", "#e66767"]
_ERR_DARK_INK = ["#0b0b0b", "#0b0b0b"] + ["#f0efec"] * 4 + ["#0b0b0b"]


def _heat_vars(fills, inks, prefix: str = "h") -> str:
    return " ".join(f"--{prefix}{i}:{f}; --{prefix}{i}i:{k};" for i, (f, k) in enumerate(zip(fills, inks)))


_DARK = ("color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; "
         "--muted:#898781; --line:#2c2c2a; --border:rgba(255,255,255,.10); --accent:#3987e5; --accent-text:#6da7ec; "
         "--accent2:#d95926; --good:#0ca30c; --warn:#fab219; --bad:#d03b3b; "
         + _heat_vars(_HEAT_DARK, _HEAT_DARK_INK) + " " + _heat_vars(_ERR_DARK, _ERR_DARK_INK, "e"))

CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
        --muted:#898781; --line:#e1e0d9; --border:rgba(11,11,11,.10); --accent:#2a78d6; --accent-text:#256abf;
        --accent2:#eb6834; --good:#006300; --warn:#8a5a00; --bad:#d03b3b; """ + _heat_vars(_HEAT_LIGHT, _HEAT_LIGHT_INK) + " " + _heat_vars(_ERR_LIGHT, _ERR_LIGHT_INK, "e") + """ }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { """ + _DARK + """ } }
:root[data-theme="dark"] { """ + _DARK + """ }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width:1240px; margin:0 auto; padding:24px 16px 64px; }
header { display:flex; flex-wrap:wrap; gap:12px; align-items:flex-end; justify-content:space-between; }
h1 { font-size:26px; margin:0; font-weight:600; letter-spacing:-.01em; }
.sub { color:var(--ink2); margin:4px 0 0; }
button { font:inherit; font-size:13px; color:var(--ink2); background:var(--surface);
         border:1px solid var(--border); border-radius:8px; padding:6px 12px; cursor:pointer; }
button:hover { color:var(--ink); }
[hidden] { display:none !important; }
.tabs { position:sticky; top:0; z-index:5; display:flex; gap:2px; overflow-x:auto; margin:16px -16px 0;
        padding:8px 16px 0; background:var(--page); border-bottom:1px solid var(--line); scrollbar-width:thin; }
.tabs a { flex:0 0 auto; padding:8px 12px; color:var(--ink2); text-decoration:none; font-size:14px;
          border-bottom:2px solid transparent; margin-bottom:-1px; white-space:nowrap; }
.tabs a:hover { color:var(--ink); }
.tabs a[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--accent); font-weight:600; }
.subnav { display:flex; flex-wrap:wrap; gap:4px 16px; margin:14px 0 4px; font-size:13px; }
.subnav a { color:var(--ink2); text-decoration:none; } .subnav a:hover { color:var(--accent); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin:16px 0 4px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 16px; min-width:0; }
.tile .label { color:var(--ink2); font-size:13px; }
.tile .value { font-size:28px; font-weight:600; line-height:1.2; overflow-wrap:anywhere; }
.tile .note { color:var(--muted); font-size:12px; margin:0; }
section { margin-top:36px; scroll-margin-top:56px; }
h2 { font-size:20px; font-weight:600; margin:0 0 4px; }
h3 { font-size:15px; font-weight:600; margin:16px 0 4px; }
.intro { color:var(--ink2); margin:0 0 12px; max-width:880px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:8px; margin:12px 0; min-width:0; }
.card > h3, .card > p, .card > .kv, .card > ul, .card > .formula { margin-left:10px; margin-right:10px; }
.pair { display:grid; grid-template-columns:minmax(0, 3fr) minmax(0, 2fr); gap:12px; }
.pair > div { min-width:0; }
.halves { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap:12px; }
.halves > div { min-width:0; }
@media (max-width:900px) { .pair { grid-template-columns:minmax(0, 1fr); } }
.chart { width:100%; min-height:280px; }
details { margin:0 4px 12px; }
summary { cursor:pointer; color:var(--ink2); font-size:14px; padding:4px 0; }
.tbl-wrap { overflow-x:auto; }
.tbl-wrap.tall { max-height:560px; overflow-y:auto; }
table.tbl { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
.tbl th, .tbl td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
.tbl th { color:var(--ink2); font-weight:600; background:var(--surface); position:sticky; top:0; }
.tbl td[data-v] { text-align:right; }
table.sortable th { cursor:pointer; user-select:none; }
table.sortable th::after { content:" \\2195"; color:var(--muted); font-weight:400; }
table.sortable th[aria-sort="ascending"]::after { content:" \\2191"; color:var(--ink2); }
table.sortable th[aria-sort="descending"]::after { content:" \\2193"; color:var(--ink2); }
""" + "".join(f"td.h{i} {{ background:var(--h{i}); color:var(--h{i}i); }}\n" for i in range(HEAT_STEPS)) + "".join(
    f"td.e{i}, span.e{i} {{ background:var(--e{i}); color:var(--e{i}i); }}\n" for i in range(HEAT_STEPS)) + """
.legend { display:flex; flex-wrap:wrap; gap:4px; margin:6px 10px; font-size:12px; color:var(--ink2); align-items:center; }
.legend span[class^="e"] { padding:1px 8px; border-radius:4px; }
tr.mine td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
td small { display:block; font-size:11px; opacity:.8; }
.tbl .cell-wrap { white-space:normal; min-width:260px; max-width:640px; }
p.note { color:var(--ink2); font-size:13px; margin:8px 10px; }
a { color:var(--accent-text); }
.verdict { display:flex; flex-wrap:wrap; gap:8px 24px; align-items:flex-start; padding:10px 10px 4px; }
.verdict .value { font-size:48px; font-weight:600; line-height:1; }
.verdict .value span { font-size:18px; font-weight:400; color:var(--muted); }
.verdict > div:last-child { flex:1 1 320px; min-width:0; }
.verdict p { margin:0 0 6px; }
p.big { font-size:18px; font-weight:600; }
.card h3 { font-size:15px; font-weight:600; margin:14px 10px 4px; }
.card ul { margin:4px 0 8px; padding-left:28px; }
h2.stage { display:flex; flex-direction:column; gap:2px; margin:0 0 6px; }
.stage-n { font-size:12px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--accent-text); }
.stage-t { font-size:20px; font-weight:600; }
.kv { display:flex; flex-wrap:wrap; gap:6px 22px; margin:8px 10px; }
.kv > div { display:flex; flex-direction:column; min-width:0; }
.kv .k { color:var(--muted); font-size:12px; }
.kv .v { font-size:16px; font-weight:600; font-variant-numeric:tabular-nums; }
.formula { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; background:var(--page);
           border:1px solid var(--line); border-radius:8px; padding:10px 12px; overflow-x:auto; white-space:pre-wrap; color:var(--ink); }
.pill { display:inline-block; font-size:12px; line-height:1.4; padding:1px 8px; border-radius:999px;
        border:1px solid var(--border); color:var(--ink2); }
.pill.good { color:var(--good); } .pill.warn { color:var(--warn); } .pill.bad { color:var(--bad); }
.filters { display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; margin:8px 10px; font-size:13px; color:var(--ink2); }
.filters input, .filters select { font:inherit; font-size:13px; color:var(--ink); background:var(--page);
        border:1px solid var(--border); border-radius:8px; padding:5px 8px; }
.filters input[type=search] { width:min(220px, 100%); }
.filters input[type=number] { width:80px; }
.bars { display:flex; gap:4px; align-items:flex-end; overflow-x:auto; padding:24px 10px 4px; }
.bars .col { flex:1 0 34px; max-width:64px; display:flex; flex-direction:column; align-items:stretch; }
.bars .bar-area { height:120px; display:flex; align-items:flex-end; }
.bars .bar { width:100%; height:var(--h); min-height:2px; background:var(--accent); border-radius:4px 4px 0 0;
             position:relative; }
.bars .bar span { position:absolute; top:-18px; left:0; right:0; text-align:center; font-size:11px; color:var(--ink2); }
.bars .sub { height:26px; display:flex; flex-direction:column; justify-content:flex-start; margin-top:3px; }
.bars .sub i { display:block; height:6px; width:var(--s); background:var(--muted); border-radius:3px; }
.bars .sub b { font-weight:400; font-size:10px; color:var(--muted); }
.bars .lbl { font-size:11px; color:var(--muted); text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.explorer-pick { display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; margin:4px 0 8px; }
.explorer-pick select { font:inherit; font-size:14px; color:var(--ink); background:var(--surface);
        border:1px solid var(--border); border-radius:8px; padding:6px 10px; max-width:100%; }
#gloss-filter { font:inherit; font-size:14px; color:var(--ink); background:var(--page);
                border:1px solid var(--border); border-radius:8px; padding:6px 10px;
                margin:10px 10px 4px; width:min(360px, calc(100% - 20px)); }
.gloss { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 320px), 1fr));
         gap:0 28px; margin:0 10px 8px; }
.gloss dt { font-weight:600; }
.gloss dd { margin:2px 0 12px; color:var(--ink2); font-size:14px; }
.term:target dt { color:var(--accent); }
footer { color:var(--muted); font-size:12px; margin-top:48px; }
"""

JS = """
(function () {
  var FIGS = JSON.parse(document.getElementById('fig-data').textContent);
  var media = window.matchMedia('(prefers-color-scheme: dark)');
  var btn = document.getElementById('theme-toggle');
  var NEXT = {auto: 'light', light: 'dark', dark: 'auto'};
  var choice = 'auto', drawn = {};
  try { choice = localStorage.getItem('fpl-theme') || 'auto'; } catch (e) {}
  if (!NEXT[choice]) { choice = 'auto'; }
  function mode() { return choice === 'auto' ? (media.matches ? 'dark' : 'light') : choice; }
  function drawIn(panel) {
    var m = mode();
    panel.querySelectorAll('.chart[id^="fig-"]').forEach(function (el) {
      var id = el.id.slice(4);
      if (!FIGS[id] || drawn[id] === m || el.offsetParent === null) { return; }
      // the box takes the figure's own height (a window resize makes Plotly fit the plot
      // to its box), and Plotly gets a copy, so it can never edit the figure we keep
      var fig = JSON.parse(JSON.stringify(FIGS[id][m]));
      if (fig.layout && fig.layout.height) { el.style.height = fig.layout.height + 'px'; }
      Plotly.react(el, fig.data, fig.layout, {responsive: true, displaylogo: false});
      drawn[id] = m;
    });
  }
  function applyTheme() {
    if (choice === 'auto') { document.documentElement.removeAttribute('data-theme'); }
    else { document.documentElement.setAttribute('data-theme', choice); }
    btn.textContent = 'Theme: ' + choice;
    document.querySelectorAll('.tab-panel:not([hidden])').forEach(drawIn);
  }
  btn.addEventListener('click', function () {
    choice = NEXT[choice];
    try { localStorage.setItem('fpl-theme', choice); } catch (e) {}
    applyTheme();
  });
  media.addEventListener('change', function () { if (choice === 'auto') { applyTheme(); } });

  // tabs: one panel at a time; the URL hash picks the tab (or any element inside it)
  var panels = Array.prototype.slice.call(document.querySelectorAll('.tab-panel'));
  var links = document.querySelectorAll('.tabs a');
  function show(name, scrollTo) {
    panels.forEach(function (p) { p.hidden = p.dataset.tab !== name; });
    links.forEach(function (a) { a.setAttribute('aria-selected', a.dataset.tab === name ? 'true' : 'false'); });
    var panel = document.getElementById('tab-' + name);
    if (panel) { drawIn(panel); }
    if (scrollTo) { scrollTo.scrollIntoView(); }
  }
  function route() {
    var id = decodeURIComponent((location.hash || '').slice(1));
    var el = id ? document.getElementById(id) : null;
    if (el && el.classList.contains('tab-panel')) { show(el.dataset.tab); window.scrollTo(0, 0); return; }
    var panel = el ? el.closest('.tab-panel') : null;
    if (panel) { show(panel.dataset.tab, el); return; }
    if (panels.length) { show(panels[0].dataset.tab); }
  }
  window.addEventListener('hashchange', route);
  route();
  applyTheme();

  // sortable tables: click a header; numbers sort by data-v, text alphabetically
  document.querySelectorAll('table.sortable').forEach(function (table) {
    var heads = table.querySelectorAll('thead th');
    heads.forEach(function (th, col) {
      th.addEventListener('click', function () {
        var asc = th.getAttribute('aria-sort') === 'descending';
        heads.forEach(function (h) { h.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
        var body = table.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = a.cells[col], y = b.cells[col];
          var xv = x.dataset.v, yv = y.dataset.v, r;
          if (xv !== undefined && yv !== undefined) { r = parseFloat(xv) - parseFloat(yv); }
          else if (xv !== undefined) { r = 1; } else if (yv !== undefined) { r = -1; }
          else { r = x.textContent.localeCompare(y.textContent); }
          return asc ? r : -r;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
  });

  // filters: <div class="filters" data-table="id"> with inputs named by data-filter
  document.querySelectorAll('.filters[data-table]').forEach(function (bar) {
    var table = document.getElementById(bar.dataset.table);
    if (!table) { return; }
    var inputs = bar.querySelectorAll('[data-filter]'), count = bar.querySelector('.count');
    function apply() {
      var f = {};
      inputs.forEach(function (i) { f[i.dataset.filter] = i.value.trim().toLowerCase(); });
      var shown = 0;
      Array.prototype.forEach.call(table.tBodies[0].rows, function (r) {
        var ok = (!f.text || r.textContent.toLowerCase().indexOf(f.text) !== -1)
              && (!f.pos || r.dataset.pos === f.pos.toUpperCase())
              && (!f.club || r.dataset.club === f.club)
              && (!f.maxprice || parseFloat(r.dataset.price) <= parseFloat(f.maxprice))
              && (!f.maxown || parseFloat(r.dataset.own) <= parseFloat(f.maxown));
        r.hidden = !ok;
        if (ok) { shown += 1; }
      });
      if (count) { count.textContent = shown + ' shown'; }
    }
    inputs.forEach(function (i) { i.addEventListener('input', apply); });
    apply();
  });

  // player explorer: one panel per player, picked from a select
  document.querySelectorAll('select[data-explorer]').forEach(function (sel) {
    var root = document.getElementById(sel.dataset.explorer);
    function pick() {
      root.querySelectorAll('.explore').forEach(function (p) { p.hidden = p.dataset.player !== sel.value; });
    }
    sel.addEventListener('change', pick);
    pick();
  });

  // glossary filter
  var box = document.getElementById('gloss-filter');
  if (box) {
    box.addEventListener('input', function () {
      var q = box.value.trim().toLowerCase(), any = false;
      document.querySelectorAll('.gloss-group').forEach(function (g) {
        var shown = 0;
        g.querySelectorAll('.term').forEach(function (t) {
          var hit = !q || t.textContent.toLowerCase().indexOf(q) !== -1;
          t.hidden = !hit;
          if (hit) { shown += 1; }
        });
        g.hidden = shown === 0;
        if (shown) { any = true; }
      });
      document.getElementById('gloss-none').hidden = any;
    });
  }
})();
"""


def render(r: Report) -> str:
    tiles = "".join(f'<div class="tile"><div class="label">{esc(a)}</div><div class="value">{esc(b)}</div>'
                    f'<p class="note">{esc(c)}</p></div>' for a, b, c in r.tiles)
    known = [t for t, _ in TABS]
    used = [(t, title) for t, title in TABS if any(s.tab == t for s in r.sections)]
    used += [(t, t.title()) for t in dict.fromkeys(s.tab for s in r.sections) if t not in known]
    nav = "".join(f'<a href="#tab-{esc(t)}" data-tab="{esc(t)}" role="tab">{esc(title)}</a>' for t, title in used)
    panels = []
    for k, (t, _) in enumerate(used):
        secs = [s for s in r.sections if s.tab == t]
        sub = ("<nav class=\"subnav\">" + "".join(f'<a href="#{esc(s.id)}">{esc(s.title)}</a>' for s in secs)
               + "</nav>") if len(secs) > 1 else ""
        if k == 0 and tiles:                              # the overview tiles open the first tab
            sub = f'<div class="tiles">{tiles}</div>' + sub
        body = "".join(
            f'<section id="{esc(s.id)}">'
            + ("" if s.blocks and s.blocks[0].startswith("<h2") else f"<h2>{esc(s.title)}</h2>")
            + (f'<p class="intro">{esc(s.intro)}</p>' if s.intro else "") + "".join(s.blocks) + "</section>"
            for s in secs)
        panels.append(f'<div class="tab-panel" id="tab-{esc(t)}" data-tab="{esc(t)}" role="tabpanel">{sub}{body}</div>')
    # allow_nan=False: a stray NaN would make the page's JSON.parse fail, so fail here instead
    figures = json.dumps(r.figures, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(r.title)}</title>
<style>{CSS}</style>
<script>{get_plotlyjs()}</script>
</head>
<body>
<div class="wrap">
<header><div><h1>{esc(r.title)}</h1><p class="sub">{esc(r.subtitle)}</p></div>
<button id="theme-toggle" type="button">Theme: auto</button></header>
<nav class="tabs" role="tablist">{nav}</nav>
{''.join(panels)}
<footer>{esc(r.footer)}</footer>
</div>
<script type="application/json" id="fig-data">{figures}</script>
<script>{JS}</script>
</body>
</html>
"""
