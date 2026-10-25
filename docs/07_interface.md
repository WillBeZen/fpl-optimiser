# 07 — Interface: the weekly run and the scout's dashboard

**Goal:** one command each gameweek, and one page that shows everything: what to do
this week, how confident to be, what each of your players should do in each fixture,
and every stage of the maths behind it.

```bash
python scripts/run_week.py --open
```

| step | what happens |
|---|---|
| 1 | refresh FPL data into the season store, and check `config.py`'s rules against FPL's |
| 2 | fit the models on every filed season; expected points for the next gameweeks (the blend of the points and event models) |
| 3 | the best squad from scratch (what a wildcard would build) |
| 4 | **your transfer plan** with its chips, if `FPL_TEAM_ID` is set, plus the what-ifs (0..n transfers, more money, each chip in each gameweek) |
| 5 | **Claude's news check** on the plan, if `ANTHROPIC_API_KEY` is set |
| 6 | the walk-forward backtest of every model; the accuracy check of the last 5 finished gameweeks (with the half-life check and its reconciliation, printed too); then `reports/fpl_report.html` and the headline in the terminal |

A full run takes about 5–6 minutes (the backtest and the what-ifs are most of it; the
plan with chips about a minute and a half).

Options: `--no-pull` (use the data already filed), `--team-id N`, `--free-transfers N`
(FPL doesn't publish them; the estimate comes from your history), `--horizon N`,
`--quick` (skip the backtest and the what-ifs, about 2 minutes faster), `--no-advisor`,
`--advisor-effort low|medium|high|xhigh|max` and `--max-searches N` (a cheaper or deeper
news check; the terminal prints its tokens and cost), `--open`.

**News checks are kept.** A successful check is saved to `reports/last_advice.json`;
`--no-advisor` reuses it (marked with when it ran) as long as it's for the same
gameweek, so re-running the report doesn't pay for another.

**Setup, once:** put your team id (the number in your team's URL:
`fantasy.premierleague.com/entry/<id>/event/<gw>`) and your Anthropic key in `.env`:
```
FPL_TEAM_ID=1234567
ANTHROPIC_API_KEY=sk-ant-...
```
Without them the run still works: no personal plan, and no news check.

## The dashboard (`reports/fpl_report.html`)
One self-contained file (~6 MB, Plotly embedded): no server, no internet. It's
generated output, git-ignored, so rebuild it each week. Tabs along the top (the URL
remembers which, so links like `#p-ladder` jump straight to a section); every table
sorts by clicking a column; light and dark themes.

**1 · This week.** Tiles (deadline, the plan's gain, transfers, captain, news-check
score), this week's moves with what each gains per gameweek and the money it frees,
the plan against no transfers, your team sheet, **alerts** (injuries, price changes
FPL says are close, blank and double gameweeks, rotation risk), Claude's news check,
and the captain options with their chances of a goal or assist and ownership.

**2 · My squad.** Your 15 plus this week's buys:
- expected points per gameweek with the opponent under each number (heat-shaded);
- the next gameweek **event by event**: expected minutes, P(plays), P(60+), xG, xA,
  P(goal), P(assist), P(clean sheet), P(defensive contribution), saves, bonus, and
  the event model, points model and blended expected points side by side;
- **where the points come from** (appearance / attack / defence / bonus and cards);
- every fixture in the horizon;
- **against every club:** each player's expected points against each of the 20 clubs;
- a **player explorer** (your squad, then the top 40): status, set pieces, FPL's own
  estimate, fixtures event by event, and his **form story**: each match's points with
  its recency weight, the form the model sees against his plain average and last
  season, and his form under a shorter or longer half-life.

**3 · Fixtures.** Team ratings (attack vs defence chart and table), the **fixture
ticker** for goals and for clean sheets over the horizon (sorted, heat-shaded, with
each opponent), and the defensive styles the points model uses.

**4 · Scout.** Captaincy across the game; the **player market** (top 300, filter by
position, club, max price and max ownership; your players marked); the best three per
position and price band; differentials (under 10% owned); price watch (net transfers
this gameweek and FPL's progress to a price change); set-piece takers; where this model
and FPL's own estimate disagree; and finishing above or below xG.

**5 · The model** — the prediction engine in ten stages: the data; recency weighting
(the weight of every past match and last season, how last season fades, a worked
example for your captain, your squad's form under other half-lives); team
characteristics; the inputs; **player types against defensive styles** (what each type
has actually scored against tight, middling and leaky defences) and how the points
model responds to the opponent; the minutes and points models and what they lean on
(permutation importance); the event model's formulas and average points by source; the
blend and FPL's flags (who they cut); and the walk-forward backtest of every method.

**6 · Accuracy** — every finished gameweek replayed as it stood at its deadline (models,
features, players, prices and the planner's pool from before it only):
- every method side by side, for the planner's pool and for every player: the blend,
  each model alone, and four simple baselines;
- **whether the blend beats each one**, gameweek by gameweek with 90% intervals;
- each method gameweek by gameweek, and the blend's walk-forward weight;
- every pool player's predicted and actual points over the last `ACCURACY_WINDOW_GW`
  gameweeks, shaded by over- or under-prediction;
- player totals, calibration, and accuracy by position and by scoring source;
- the biggest misses, the half-life check, and the logged deadline forecasts;
- **data checks**, kept separate because they test the data, not the model.

See `docs/08_accuracy.md`.

**7 · Decision test** — do the predictions pick better teams? A legal team every week
from each method's predictions, scored on real points with FPL's rules (substitutes,
vice-captain), against the hindsight-best team. Then a season with transfers and hits
per method, next to keeping the first squad and your real team. See
`docs/09_decision_test.md`.

**8 · The planner** — the integer program in twelve stages, with your numbers: inputs
(bought for / now / sells for, bank, free transfers, dials); the candidate pool; the
problem's variables, rules and objective, and its size, solve time and whether it was
proven optimal; the no-transfer baseline; **how many transfers** (each extra move's
worth, including whether a hit pays); the plan week by week (free transfers carried,
chip, bank); **chips** (sets left, when the plan plays them, each chip in its best week
and a chip × gameweek grid against the thresholds); **why these moves** and the
alternatives; what's binding (clubs at the limit, formation) and **what more money would
buy**; the objective piece by piece; captain and bench each week; and the wildcard squad
for comparison.

**9 · Player analysis.** Player types per position, attacking threat, risers and
fallers between seasons, value, and how this season links to last (Sections 1–2).

**10 · Glossary.** About 95 terms in plain English, with a filter box. Each term has an
anchor (`#g-<slug>`, e.g. `#g-hit-threshold`); numbers in the definitions are read from
`config.py`, so they always match the settings the report was built with.

## How it's built (`fpl_optimiser/app/`)
- `html.py`: the building blocks. `Section` (with its tab), `Report`, `render()`, the
  page's styles and script (tabs, sorting, filters, explorer, glossary filter, lazy
  chart drawing per tab), and fragments: `data_table` (sortable; heat cells from one
  validated blue ramp per theme, with text contrast of at least 5.4:1; cells can be a
  (number, html) pair; hidden columns for row attributes), `stage`, `kv`, `bars` (a
  small CSS bar chart), `pill`, `card`, `note`, `link`.
- `common.py`: `WeekResults` (what the run produced) and `Context` (the filed season),
  plus shared lookups.
- `tab_week.py`, `tab_squad.py`, `tab_fixtures.py`, `tab_scout.py`, `tab_model.py`,
  `tab_accuracy.py`, `tab_decisions.py`, `tab_planner.py`: one `add(report, week, ctx)` each. Each tab skips what its inputs
  don't support, so the page always builds.
- `charts.py`: pure Plotly builders, light *and* dark, from a validated
  colour-blind-safe palette (three colours for scatters, four for stacked bars and
  lines, one blue ramp for heatmaps). Heatmap labels are added in one list (adding
  them one at a time is quadratic in Plotly).
- `report.py`: `build_report(season, week=WeekResults(...))` loads the season, builds
  the context, adds this week's tabs, the analysis tab and the glossary.
- `glossary.py`: the terms and definitions.

Untrusted text (player names, FPL news, everything the advisor returns) is escaped,
and links are `http(s)` only. DOM ids are unique (a duplicate breaks the page script;
a test checks it).

## Tests
`tests/test_report.py` (the Accuracy tab's own tests: `tests/test_accuracy.py`): charts use their theme's palette; the page is self-contained;
chart ids can't collide with section ids, and no DOM id is used twice; hostile names
and advice stay escaped and `javascript:` links are dropped; every tab's sections
appear with a plan and news check, and degrade gracefully without a team, the advisor
or season data; the full dashboard builds from a filed store (forecast, events,
backtest, explorer); tables sort, shade and escape; the glossary's terms and anchors are
unique and quote the live `config.py` settings.
