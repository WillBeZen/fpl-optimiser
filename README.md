# FPL optimiser

A two-engine Fantasy Premier League decision system:

1. a **prediction model** estimates each player's expected points (recency-weighted,
   with opponents modelled by *characteristics* rather than as 20 separate teams,
   and last season carried in as a prior), and
2. a **multi-week integer program** picks the squad, captain and transfers that
   maximise those points under FPL's rules.

A **Claude advisor** then interprets the result against live football news and
returns a 0–100 "how strongly to act" score. Everything is presented as one
self-contained **HTML report** you open locally.

Full plan and rationale: [`docs/00_charter.md`](docs/00_charter.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your ANTHROPIC_API_KEY
```

> **macOS + iCloud:** don't keep the repo (or its venv) in an iCloud-synced folder
> (Desktop or Documents with "Desktop & Documents Folders" on). iCloud offloads the
> venv's files to the cloud and every `import` stalls while they download, so
> `pytest` looks frozen. Keep the repo somewhere like `~/Developer`.

## Check it works

```bash
python -m pytest -q                # offline test suite, a few seconds
python config.py                   # prints "config OK"
```

## Use it

```bash
python scripts/pull_fpl.py             # file the live FPL season into data/raw/fpl/<season>/
python scripts/build_features.py       # training table across seasons -> data/processed/
python scripts/make_report.py --open   # reports/fpl_report.html (works offline)
```

## Layout

```
config.py                  all FPL rules + tunable dials  (start here)
fpl_optimiser/
  ingest/                  Section 1 — FPL API, season store, cross-season links, opponent stats
  features/                Section 2 — recency, opponent archetypes, training table, last-season
                           prior, multi-season table, player types
  model/                   Section 3 — minutes & points prediction, backtest
  optimise/                Sections 4–5 — squad MILP + multi-week planner
  claude/                  Section 6 — Claude advisor
  app/                     Section 7 — charts + the HTML report
scripts/                   pull_fpl, build_features, make_report, verify_setup
docs/                      one plain-English write-up per section
tests/                     offline: mock payloads and tiny synthetic seasons, never the network
data/                      raw (filed seasons) / processed / cache  (git-ignored)
reports/                   the generated HTML report  (git-ignored)
```

## Build status

Sections 0–2 are done; the HTML report (Section 7) is started. Each section adds its
module **and** its doc, ticked off in the charter as we go.

## Running data pulls

Live data comes from `fantasy.premierleague.com`, which you call from your own
machine. `pull_fpl.py` is polite by design: one keep-alive connection, a short
pause between player requests, retries that back off, and re-runs that skip files
fetched in the last 12 hours. A finished season is frozen and never re-pulled. If
you switch the opponent source to FBref, `soccerdata` caches and rate-limits its
requests: don't disable that.
