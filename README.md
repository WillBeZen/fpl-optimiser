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

## Every gameweek

```bash
python scripts/run_week.py --open
```

That one command refreshes FPL's data, predicts the next gameweeks, plans your
transfers (bank, spend, or take a −4 only if it clearly pays), asks Claude to check
the news, re-checks the models against the season so far, and opens the dashboard
(about 3–4 minutes). `python scripts/run_week.py --help` lists the
options (e.g. `--free-transfers 2` if the estimate is off,
`--advisor-effort low --max-searches 2` for a cheaper news check, `--quick` to skip the
backtest and what-ifs, `--no-advisor` to reuse this gameweek's last news check instead
of paying for a new one). The dashboard ends with a glossary of every term it uses.

## Check it works

```bash
python -m pytest -q                # 114 offline tests, ~40 seconds
python config.py                   # prints "config OK"
python scripts/check_claude.py     # one tiny Claude request: is your key working?
```

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
scripts/                   run_week (weekly), pull_fpl, build_features, make_report, verify_setup
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
