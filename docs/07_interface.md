# 07 — Interface: the HTML report

**Goal:** one page you open locally that shows what the system knows: player
types, form, value and the season link now; and later the predictions, the
optimiser's squad, transfers and captain, and Claude's advice.

```bash
python scripts/pull_fpl.py            # optional: refresh the live season first
python scripts/make_report.py --open  # -> reports/fpl_report.html, opened in your browser
```

**One self-contained file.** `reports/fpl_report.html` (~5 MB) has Plotly's
JavaScript embedded, so it needs no server and no internet: double-click it, keep
it, share it. It is generated output, so it is git-ignored; rebuild it whenever the
data changes. (Streamlit was the original plan; a static file is simpler and
nothing has to keep running.)

## What's on the page
- **Summary tiles:** gameweeks played, players, how many link to last season, new
  arrivals, and the link check against FPL's own history.
- **Player types**, position by position: each profiled player on the two stats
  that best show the types (e.g. defensive contributions vs xGI for midfielders),
  coloured by type, beside a "fingerprint" heatmap showing what sets each type apart
  (in standard deviations from the position average).
- **Attacking threat:** xG/90 against xA/90 for midfielders and forwards.
- **Risers and fallers:** the same players' points per 90 last season against this
  season, with the biggest movers labelled.
- **Value:** the best points per £m so far.
- **The season link:** new arrivals by club, how players were linked, and which
  clubs went up and down.
- **Coming next:** placeholders for Sections 3–6.

## How it's built (`fpl_optimiser/app/`)
- `charts.py`: pure Plotly builders, DataFrames in and figures out. Each takes a
  `theme`, and the page carries every chart in light *and* dark and draws whichever
  matches the viewer's setting (with a toggle). Colours come from a validated
  colour-blind-safe palette: at most three categorical colours per scatter, one
  colour for single series, and a blue ↔ red diverging scale with a neutral
  midpoint for z-scores. Marks are thin and grids are hairlines. Only a few points
  are labelled directly; the hover tooltips carry the detail.
- `report.py`: `build_report()` gathers the data, and `Report.write()` renders the
  page. Every chart has a collapsible table under it, so every value is readable
  without hovering. Player names are treated as data: escaped in tables, and kept
  inside a JSON block they can't break out of.
- **Adding a section** (Sections 3–6): build a `Section(id, title, intro, blocks)`,
  where blocks are `card(r.chart("my-id", my_builder, df))` and/or `table(df, ...)`,
  and append it to `report.sections`.

## Tests
`tests/test_report.py`: charts use their theme's validated palette with at most
three colours per scatter, and the diverging scale is centred; the page is
self-contained (no external scripts); a chart can never collide with a section's
id; hostile names stay escaped and can't break the embedded data; and the whole
report builds from a filed store.
