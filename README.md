# FPL optimiser

A two-engine Fantasy Premier League decision system:

1. a **prediction model** estimates each player's expected points (recency-weighted,
   with opponents modelled by *characteristics* rather than as 20 separate teams,
   and last season carried in as a prior), and
2. a **multi-week integer program** picks the squad, captain and transfers that
   maximise those points under FPL's rules.

A **Claude advisor** then checks the plan against live football news and returns a
0–100 "how strongly to act" score. Everything lands in one self-contained **HTML
dashboard** you open locally: this week's plan, what each of your players should do in
each fixture, fixture tickers, a scout's view of the whole market, and every stage of
the model and the optimiser, explained with your numbers.

Full plan and rationale: [`docs/00_charter.md`](docs/00_charter.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then fill in the two lines below
```

`.env` (git-ignored, never commit it):
```
FPL_TEAM_ID=1234567                # the number in your team's URL on the FPL site
ANTHROPIC_API_KEY=sk-ant-...       # for the Claude news check
```
Both are optional: without them you still get predictions and the best squad from
scratch, just no personal transfer plan or news check.

> **macOS + iCloud:** don't keep the repo (or its venv) in an iCloud-synced folder
> (Desktop or Documents with "Desktop & Documents Folders" on). iCloud offloads the
> venv's files to the cloud and every `import` stalls while they download, so
> `pytest` looks frozen. Keep the repo somewhere like `~/Developer`.

## How to run it

### First time
1. Set up as above (Python 3.11+; tested on 3.13), then check everything works:
   ```bash
   python -m pytest -q              # 150 offline tests, about three minutes; never touches the network
   python config.py                 # prints "config OK"
   python scripts/verify_setup.py   # live check: FPL reachable, opponent source, rules match FPL's
   python scripts/check_claude.py   # optional: one tiny Claude request (a fraction of a cent)
   ```
2. File this season's data (about 700 polite requests, a few minutes the first time):
   ```bash
   python scripts/pull_fpl.py
   ```

### Every gameweek (before the deadline)
```bash
source .venv/bin/activate
python scripts/run_week.py --open
```
This one command, in about 3–5 minutes:
1. refreshes FPL's data and checks `config.py`'s rules against FPL's published ones;
2. fits the models and predicts every player's expected points for the next
   `PLANNING_HORIZON_GW` gameweeks (a blend of the points model and the event model);
3. picks the best squad from scratch (the wildcard yardstick);
4. plans **your** transfers, captain and chips (if `FPL_TEAM_ID` is set), bank or
   spend, a −4 only if it clearly pays, plus what-ifs (0..n transfers, more money);
5. asks Claude to check the news on the plan (if `ANTHROPIC_API_KEY` is set; costs a
   few cents, and the result is saved for reuse that gameweek);
6. backtests the models on the season so far, checks how the last 5 finished gameweeks'
   predictions lined up with what players scored (printing the reconciliation), writes
   `reports/fpl_report.html`, prints the headline moves, and opens the dashboard. It also
   files this week's forecast, so its accuracy can be checked once the gameweek is played.

Then make the moves on the FPL site yourself: the system never touches your team.

| option | what it does |
|---|---|
| `--open` | open the dashboard in your browser when done |
| `--no-pull` | use the data already filed (skip the refresh) |
| `--quick` | skip the backtest and the planner's what-ifs (about 2 minutes faster) |
| `--no-advisor` | no new Claude check; reuses this gameweek's last saved one, if any (no cost) |
| `--advisor-effort low\|medium\|high\|xhigh\|max` | how hard Claude thinks (lower = cheaper) |
| `--max-searches N` | most web searches Claude may run ($0.01 each) |
| `--team-id N` | plan for this team id instead of `FPL_TEAM_ID` |
| `--free-transfers N` | override the free transfers estimated from your history |
| `--horizon N` | gameweeks to plan over (default `PLANNING_HORIZON_GW` in `config.py`) |

Common combinations:
```bash
python scripts/run_week.py --no-pull --no-advisor --quick --open      # fast look, no cost
python scripts/run_week.py --advisor-effort low --max-searches 2 --open   # cheap news check
```

### The dashboard
`reports/fpl_report.html` is one self-contained file: open it in any browser, no
server or internet needed (it's git-ignored; rebuild it each week). Tabs: **This week**
(moves, alerts, news check, captain), **My squad** (each player fixture by fixture,
event by event, and a player explorer), **Fixtures** (team ratings and tickers),
**Scout** (the market, targets, differentials, price watch, set pieces), **The model**
(every stage, with your numbers), **Accuracy** (every past gameweek replayed as it stood
at its deadline: the model against simple baselines with gameweek-by-gameweek
uncertainty, predicted vs actual for every player the planner considers, and data
checks), **Decision test** (the teams, captains and transfers each method would have
picked, scored on real points against the hindsight-best team and your own), **The
planner** (every stage), **Player analysis** and a **Glossary**. Every table sorts by clicking a column; light and dark themes.

### Other scripts
```bash
python scripts/pull_fpl.py                  # just refresh the filed season (--limit N for a quick test)
python scripts/build_features.py            # just build the multi-season training table
python scripts/make_report.py --open        # the analysis-only report (no plan, no Claude)
```

### Tuning
Every rule and strategy dial is in `config.py` (half-life, planning horizon, hit
threshold, horizon decay, bench weight, event-model weight, chip thresholds, ...). Change
one, re-run, and compare the backtest on the dashboard's **The model** tab.

## Layout

```
config.py                  all FPL rules + tunable dials  (start here)
fpl_optimiser/
  ingest/                  Section 1 — FPL API, season store, cross-season links,
                           your team, rules check, opponent stats
  features/                Section 2 — recency, opponent archetypes, training table,
                           last-season prior, multi-season table, player types
  model/                   Section 3 — minutes + points models, event model, backtest,
                           forecasts, explainers
  optimise/                Sections 4–5 — squad MILP, multi-week planner, pool, what-ifs
  claude/                  Section 6 — the Claude news check
  app/                     Section 7 — charts + the tabbed HTML dashboard
scripts/                   run_week (weekly), pull_fpl, build_features, make_report, verify_setup,
                           check_claude
docs/                      one plain-English write-up per section
tests/                     offline: mock payloads and tiny synthetic seasons, never the network
data/                      raw (filed seasons) / processed / cache  (git-ignored)
reports/                   the generated HTML report  (git-ignored)
```

## Build status

Sections 0–7 are built and tested, chip planning included. Next: the full 2025/26
match data from a public archive (see the charter's "What's next").

## Running data pulls

Live data comes from `fantasy.premierleague.com`, which you call from your own
machine. The pull is polite by design: one keep-alive connection, a short pause
between player requests, retries that back off, and re-runs that skip files fetched
in the last 12 hours. A finished season is frozen and never re-pulled.
