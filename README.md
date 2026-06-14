# FPL optimiser

A two-engine Fantasy Premier League decision system:

1. a **prediction model** estimates each player's expected points (recency-weighted,
   with opponents modelled by *characteristics* rather than as 20 separate teams), and
2. a **multi-week integer program** picks the squad, captain and transfers that
   maximise those points under FPL's rules.

A **Claude advisor** then interprets the result against live football news and
returns a 0–100 "how strongly to act" score.

Full plan and rationale: [`docs/00_charter.md`](docs/00_charter.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your ANTHROPIC_API_KEY
```

## Check it works

```bash
python -m pytest -q                # validates config
python config.py                   # prints "config OK"
```

## Layout

```
config.py                  all FPL rules + tunable dials  (start here)
fpl_optimiser/
  ingest/                  Section 1 — FPL API + FBref data
  features/                Section 2 — recency weighting, opponent characteristics
  model/                   Section 3 — minutes & points prediction, backtest
  optimise/                Sections 4–5 — squad MILP + multi-week planner
  claude/                  Section 6 — Claude advisor
  app/                     Section 7 — Streamlit dashboard
docs/                      one plain-English write-up per section
tests/
data/                      raw / processed / cache  (git-ignored)
```

## Build status

Section 0 (scaffold) is done. Each later section adds its module **and** its doc,
ticked off in the charter as we go.

## Running data pulls

Live data comes from `fantasy.premierleague.com` and `fbref.com`, which you call
from your own machine. Be a good citizen with FBref: `soccerdata` caches and
rate-limits requests — don't disable that.
