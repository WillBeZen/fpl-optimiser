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
(`config.PLANNING_HORIZON_GW`), linked week to week:

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

## Faster: the candidate pool (`optimise/pool.py`)
Most of the ~700 players can never be in an optimal squad: at least `quota + 5`
others at their position cost no more and score at least as much in every gameweek of
the horizon. They're dropped before solving (your own players are always kept): 669
→ ~240 players and the same plan, about 2.5× faster. `Plan.stats` records the problem
size and solve time. Free transfers carried each week (`Week.free_after`) are exact (a
tiny tie-break stops the solver under-reporting them when it doesn't matter to it).

## What-ifs (`optimise/whatif.py`)
Each re-solves the same program with one thing changed (in parallel threads; CBC runs
as its own process):

- `transfer_ladder`: the best plan with exactly 0, 1, 2, ... transfers this week (up
  to one hit), and what each extra move adds to the objective;
- `budget_value`: the plan with £0.5m / £1m / £2m more in the bank;
- `alternatives`: for each move, the next-best players who could have come in
  (same position, affordable with that slot's money, club limit kept);
- `transfer_reasons`: each move out vs in, gameweek by gameweek, with the money freed;
- `constraint_use`: clubs at the 3-player limit, money left, formation;
- `objective_parts`: the solver's objective split into each week's XI, captain and
  bench (decayed), the hit penalty and banked transfers.

`run_week.py` runs the ladder and budget what-ifs by default (`--quick` skips them).

## What isn't planned yet
Chips (wildcard, free hit, bench boost, triple captain). Prices are assumed not to
change during the horizon. Your squad, free transfers and selling prices come from FPL's public
data (`ingest/my_team.py`); `run_week.py --free-transfers N` overrides the estimate.

## Tests
`tests/test_optimise.py`: selling prices follow FPL's rule; with transfers disabled the
squad never changes and free transfers roll over (1 → 2 → 3); an injured star is
replaced using the free transfer, without a hit; and, in a controlled market, a −4 is
taken when it gains 8 points but not when it gains 5 (threshold 6). Pruning: a dear,
pointless player is dropped unless owned, and the pruned and full solves give the same
plan; free transfers carried week to week follow the rules; the ladder's best rung is
the plan and replacing an injured starter pays; more budget never hurts; alternatives
are same-position, affordable and outside the squad; objective parts add up.
