# 07 — Interface: the weekly run and the HTML report

**Goal:** one command each gameweek, and one page that shows everything: what to do
this week, how confident to be, and the analysis behind it.

```bash
python scripts/run_week.py --open
```

| step | what happens |
|---|---|
| 1 | refresh FPL data into the season store, and check `config.py`'s rules against FPL's |
| 2 | fit the prediction model on every filed season; expected points for the next 5 gameweeks |
| 3 | the best squad from scratch (what a wildcard would build) |
| 4 | **your transfer plan**, if `FPL_TEAM_ID` is set: from your squad, bank and free transfers |
| 5 | **Claude's news check** on the plan, if `ANTHROPIC_API_KEY` is set |
| 6 | write `reports/fpl_report.html` and print the headline in the terminal |

Options: `--no-pull` (use the data already filed), `--team-id N`, `--free-transfers N`
(FPL doesn't publish them; the estimate comes from your history), `--horizon N`,
`--no-advisor`, `--advisor-effort low|medium|high|xhigh|max` and `--max-searches N`
(a cheaper or deeper news check; the terminal prints its tokens and cost),
`--backtest` (re-check the model, about a minute), `--open`.

**Setup, once:** put your team id (the number in your team's URL:
`fantasy.premierleague.com/entry/<id>/event/<gw>`) and your Anthropic key in `.env`:
```
FPL_TEAM_ID=1234567
ANTHROPIC_API_KEY=sk-ant-...
```
Without them the run still works: no personal plan, and no news check.

## The report (`reports/fpl_report.html`)
One self-contained file (~5 MB, Plotly embedded): no server, no internet. It's
generated output, git-ignored, so rebuild it each week.

**This week** (top of the page):
- **Tiles:** next deadline, the plan's gain over making no transfers, transfers and
  hits this week, captain, and the news-check score.
- **This week:** expected points per gameweek with the plan against no transfers, the
  plan week by week (out / in / hits / free transfers / captain), and your team sheet.
- **News check (Claude):** score, verdict, summary, flagged players with sources,
  adjustments, risks (Section 6).
- **Best squad from scratch:** the wildcard squad.
- **Predictions:** the highest expected points next gameweek, and a table of the top
  60 over the horizon, with FPL's injury news.
- **Model check** (with `--backtest`): the walk-forward backtest against simple baselines.

**Analysis:** player types per position, attacking threat, risers and fallers between
seasons, value, and how this season links to last (Sections 1–2).

**Glossary** (last section, in the nav): about 60 terms in plain English, grouped as
the game, transfers and chips, stats, predictions, the plan, model check, news check
and seasons. A filter box narrows it as you type. Each term has an anchor
(`#g-<slug>`, e.g. `#g-hit-threshold`). Numbers in the definitions (hit threshold,
half-life, bench weight, ...) are read from `config.py`, so they always match the
settings the report was built with.

## How it's built (`fpl_optimiser/app/`)
- `charts.py`: pure Plotly builders, DataFrames in and figures out, each drawn in light
  *and* dark from a validated colour-blind-safe palette. Scatters use at most three
  colours; single series use one; the plan chart highlights the plan against a grey
  "no transfers". Marks are thin, grids are hairlines, and labels are selective.
- `report.py`: `build_report(season, week=WeekResults(...))` assembles the page.
  `_add_week` adds this week's sections; the analysis sections follow, then
  `glossary_section()`. Every chart has a table under it.
- `glossary.py`: the terms and definitions (`groups()`), and `slug()` for anchors. Untrusted text (player names, FPL news, everything the advisor
  returns) is escaped, and links are `http(s)` only.
- `scripts/run_week.py`: the weekly run; `scripts/make_report.py` rebuilds only the
  analysis part.

## Tests
`tests/test_report.py`: charts use their theme's palette, with at most three colours per
scatter; the page is self-contained (no external scripts); chart ids can't collide with
section ids; hostile names and advice stay escaped and `javascript:` links are dropped;
this week's sections appear with a plan and news check (including its cost), and
degrade gracefully without a team or the advisor; the backtest section appears when
given; the full report builds from a filed store and ends with the glossary; and
glossary terms and anchors are unique and quote the live `config.py` settings.
