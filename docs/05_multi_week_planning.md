# 05 — Multi-week planning: transfers, banking and hits

**Goal:** from *your* squad, bank and free transfers, decide what to do this week
**and** the next few: who to sell, who to buy, when to bank a transfer, and when a
−4 hit actually pays.

```python
from fpl_optimiser.optimise import planner
plan = planner.plan(forecast.xp, players, my_squad_ids, bank=0.5, free_transfers=2)
for week in plan.weeks:
    week.sold, week.bought, week.hits, week.captain, week.team
```

## One program over the whole horizon
The single-gameweek rules (Section 4) apply to every gameweek in the horizon
(`config.PLANNING_HORIZON_GW`, 5), linked week to week:

- **squad continuity:** this week's squad = last week's + bought − sold;
- **money:** bank = last week's bank + sales − purchases, never below zero. Your own
  players sell at **FPL's selling price**: you keep half of any rise, rounded down to
  £0.1m, and take any fall in full (`my_team.selling_price`);
- **free transfers:** paid transfers ≥ transfers − free ones; next week's free
  transfers ≤ this week's − used + 1, capped at 5 (`config.MAX_BANKED_TRANSFERS`).

Planning every week together is what lets it see that banking a transfer now buys two
next week, or that a hit pays back over five weeks.

## The objective, and its dials (`config.py`)
- Expected points of each week's XI, plus the captain again, plus the bench at
  `BENCH_WEIGHT` — weighted by `HORIZON_DECAY` (0.9) per week ahead, because
  predictions further out are less certain.
- **Hits** cost `HIT_THRESHOLD_POINTS` (6) in the plan, not 4. A −4 is only taken if it
  gains more than 6 expected points, a margin for prediction error, but it is
  *reported* as the real −4.
- A free transfer **still banked after the last week** is worth `FREE_TRANSFER_VALUE`
  (1.5 points); otherwise the plan would burn every transfer in its final week for any
  tiny gain, blind to the flexibility it throws away.

`max_transfers_per_gw=0` gives the **"do nothing" baseline** (best XI and captain each
week, no transfers), which is what a plan's gain is measured against.

## What isn't planned yet
Chips (wildcard, free hit, bench boost, triple captain). Prices are assumed not to
change during the horizon. Your squad, free transfers and selling prices come from FPL's public
data (`ingest/my_team.py`); `run_week.py --free-transfers N` overrides the estimate.

## Tests
`tests/test_optimise.py`: selling prices follow FPL's rule; with transfers disabled the
squad never changes and free transfers roll over (1 → 2 → 3); an injured star is
replaced using the free transfer, without a hit; and, in a controlled market, a −4 is
taken when it gains 8 points but not when it gains 5 (threshold 6).
