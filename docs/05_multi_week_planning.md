# 05 — Multi-week planning: transfers, banking, hits and chips

**Goal:** from *your* squad, bank and free transfers, decide what to do this week
**and** the next few: who to sell, who to buy, when to bank a transfer, when a
−4 hit actually pays, and when to play a chip.

```python
from fpl_optimiser.optimise import planner
plan = planner.plan(forecast.xp, players, my_squad_ids, bank=0.5, free_transfers=2,
                    chips=my_team.chips_available)       # optional: plan the chips too
for week in plan.weeks:
    week.sold, week.bought, week.hits, week.chip, week.captain, week.team
plan.chips                                               # e.g. {"bench_boost": 12}
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
week, no transfers and no chips), which is what a plan's gain is measured against.

## Chips
Which chips you still have comes from the chips you've played (`my_team.chips_available`,
from FPL's history): there are `config.CHIP_SETS_PER_SEASON` (2) sets, the first for
GW1–`CHIP_SPLIT_GW` (19, i.e. used on or before the GW19 deadline) and the second for
GW20–38; a chip played in a set's gameweeks uses that set's copy, and a set whose last
gameweek has passed has expired. The result, e.g. `{(1, 19): ("wildcard", "free_hit",
"bench_boost"), (20, 38): (all four)}`, goes to `planner.plan(chips=...)`, which adds a
yes/no decision for each chip in each gameweek it's allowed: **at most one chip a week,
each once per set, only within its set's gameweeks.**

- **Wildcard:** that week's transfers cost no hits; the new squad stays.
- **Free hit:** a different squad for one week; your own squad makes no transfers that
  week and is back the week after. Prices are held fixed in the plan, so a transfer never
  changes bank + squad value: the money for a free hit is the same whatever happens
  before it. So each week's best free-hit squad is picked up front (a one-week
  `squad.pick_squad`, your own players at their selling price) and the plan only decides
  whether to play it.
- **Bench boost:** the bench scores in full that week (instead of at `BENCH_WEIGHT`).
- **Triple captain:** the captain counts `TRIPLE_CAPTAIN_MULTIPLIER` (3) times, not 2.
- Free transfers you had are **kept** through a wildcard or free hit, plus the weekly one
  (FPL's rule since 2024/25).

**Chip thresholds** (`config.CHIP_THRESHOLD_POINTS`: wildcard 20, free hit 15, bench
boost 12, triple captain 10). A chip is played only if it gains more than its threshold,
in that gameweek's points: the threshold is decayed like the week's points. It works like
the hit threshold: the threshold is the chip's opportunity value, what it should be worth
if saved for a double or blank gameweek, so the plan doesn't burn a chip for a tiny gain.
Decaying it matters: undecayed, a double gameweek four weeks out could never earn the
bench boost, and the plan would rather spend it now for less. A chip whose set **expires
within the horizon** is worth nothing saved, so its threshold is 0 and it goes in its best
remaining week (`planner.chip_charge`).

**Keeping it fast.** Chips make the integer program much harder for CBC. Three things
keep it manageable:

1. Chip options that can't beat their threshold are dropped first. A bench boost can't
   add more than the week's best keeper and three best outfielders. A triple captain
   can't add more than the week's best player.
2. The chip effects are linked player by player (`plays`: who scores in full, 11 or all
   15; `triple`: the captain's extra), so a fractional chip in the relaxation only helps
   fractionally.
3. CBC is handed a good plan to start from. From scratch it takes minutes to find even
   the plan without chips. The steps: (1) the plan without chips; (2) keeping its squads,
   the chips worth playing (a small problem); (3) the best plan with those chips played;
   (4) the full problem, starting from that plan. CBC 2.10 misreads a starting solution
   in a maximisation, so step 4 is posed as minimising minus the objective, with CBC's
   heuristics and cuts off.

Proving that nothing beats the start can still take CBC minutes: the free hit and
wildcard loosen the relaxation by a few points. So step 4 stops after
`planner.CHIP_SOLVE_SECONDS` (60) with the best plan found, never worse than its start.
`Plan.stats["proven_optimal"]` says which, and the report shows it.

Measured on the real GW9 problem (240 candidates, 6 weeks): without chips, 16 s. With
wildcard, free hit and bench boost available (triple captain was used in GW3), 80 s, of
which step 4 is the 60 s cap: no chip clears its threshold, the answer is the no-chip
plan, unproven. With a simulated double gameweek in GW12 (six clubs' points × 1.9), the
starting plan already plays the bench boost there (+22.7 bench points), the optimum. The
first version took 267 s and found nothing, before the warm start and the
per-player links.

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
  bench (decayed; a chip's week as the chip scores it), the hit penalty, the chip
  thresholds and banked transfers;
- `chip_weeks`: each chip still available played in each gameweek it's allowed
  (alone, the rest re-planned around it) against the plan without chips. It gives the
  planner's gain, the expected points it adds, and the threshold it must beat that week.
  The report turns it into "each chip in its best week" and a chip × gameweek grid.

The ladder and budget what-ifs keep the plan's chips where the plan plays them
(`hold_chips`), so they ask only about transfers and money and need no chip decisions.
`run_week.py` runs the ladder, budget and chip what-ifs by default (`--quick` skips them).

## What isn't planned yet
Prices are assumed not to change during the horizon (which is also what makes a free
hit's squad independent of the plan). Chip thresholds are judgement calls, not fitted.
Your squad, free transfers, selling prices and chips left come from FPL's public data
(`ingest/my_team.py`); `run_week.py --free-transfers N` overrides the free-transfer
estimate.

## Tests
`tests/test_optimise.py`: selling prices follow FPL's rule; with transfers disabled the
squad never changes and free transfers roll over (1 → 2 → 3); an injured star is
replaced using the free transfer, without a hit; and, in a controlled market, a −4 is
taken when it gains 8 points but not when it gains 5 (threshold 6). Pruning: a dear,
pointless player is dropped unless owned, and the pruned and full solves give the same
plan; free transfers carried week to week follow the rules; the ladder's best rung is
the plan and replacing an injured starter pays; more budget never hurts; alternatives
are same-position, affordable and outside the squad; objective parts add up.

Chips, in controlled synthetic markets: chips left per set from the history (a used
chip, the GW19 split, an expired first set); a double gameweek (everyone's points × 2)
triggers the bench boost, an ordinary week doesn't; a blank gameweek for most of your
clubs triggers the free hit, your squad comes back untouched and free transfers are
kept; ten of your players out for good triggers a wildcard with no hits; one chip a week
(the bench boost takes the double, the triple captain the star's huge week); the split
rule (only the second set left: the bench boost waits for the double after GW19; both
sets: one in each); a chip about to expire is played for any gain; the chip grid's best
cell is the planned one, holding chips reproduces the plan, objective parts add up.
