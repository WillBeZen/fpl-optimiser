# 09 — Decision test: do the predictions pick better teams?

**Goal:** score the decisions, not just the predictions. A model can predict points
well on average and still pick no better a team than a simple rule; what matters is
the team that scores.

```bash
python scripts/run_week.py --open          # the dashboard's "Decision test" tab
```

Everything runs on the honest replay (`docs/08_accuracy.md`): at each past deadline,
every method's predictions use only what was known then, and players carry their
price at that deadline. Code: `fpl_optimiser/optimise/decisions.py`.

## FPL's scoring of a picked team (`decisions.realised`)
A team is scored on real points with FPL's real rules:
- **Automatic substitutes:** a starter who didn't play is replaced from the bench in
  order. A goalkeeper can only be replaced by the reserve goalkeeper. An outfielder
  is replaced only if the formation stays legal (at least 3 DEF, 2 MID, 1 FWD) and the
  substitute played.
- **Captain and vice:** the captain's points count twice. If the captain didn't play,
  the vice-captain's count twice instead.

## Test 1 — a legal team every week
Each gameweek, every method picks the best legal team it can see from its own
predictions: 15 players within £100m at that week's prices, at most 3 per club, a
legal XI, a captain and a vice (`optimise/squad.py`). It's a free hit every week, so it
tests picking without last week's squad as a constraint.

Reported:
- real points a gameweek (and their spread);
- what the captaincy added, and how often the captain was the XI's best scorer;
- the share of the **hindsight-best** team's points. That team is picked knowing
  every real score (the best XI and captain within the budget; its bench counts for
  nothing), so it's an upper bound for reference, not a target.

The blend is compared with each method gameweek by gameweek, paired, with a 90%
interval.

## Test 2 — a season with transfers
One squad per method, picked from scratch at the first replayed deadline. After that,
each week the planner (`optimise/planner.py`, **no chips**) decides transfers, banking
and hits from that method's multi-week predictions:
- free transfers accrue (capped at 5); a −4 hit is taken only if it clears the hit
  threshold;
- sales use FPL's selling prices (half of any rise, from the price you paid).

Reported per method:
- points after hits, transfers made, hits taken and captain points;
- **gain from transfers**: the comparison with keeping the first 15 all along (best XI
  and captain each week).

If your team id is set, your real FPL points over the same gameweeks appear alongside.
That's a reference, not like-for-like: your squad, chips and timing differ.

One path through one stretch of fixtures: early choices compound, so the season
ranking is illustrative, and the weekly test is the one with a paired comparison.

## First results (2026/27 GW2–7)
| method | weekly team, pts / GW | season with transfers, pts after hits |
|---|---|---|
| Blend (weight from earlier weeks) | 65.2 | 334 |
| Event model | 65.2 | 372 |
| Position/price × P(plays) | 63.0 | 360 |
| Points model | 48.5 | 357 |
| Recent points (EWMA) | 52.3 | 281 |
| Season average | 51.8 | 289 |
| Hindsight best | 155.0 | — |

The blend and the event model pick identical weekly teams (the walk-forward weight is
1.0 from GW3). In the season simulation they diverge from different GW2 squads.

The blend's weekly teams beat the points model's clearly: +16.7 points a gameweek, 90%
interval 3.2 to 30.2. Against position/price × P(plays) the difference is +2.2 ± 5.4,
too small to separate over six gameweeks. Every simulated season gained from transfers
(between +1 and +35 points over keeping the first squad).

## Cost and caching
The weekly test takes about 25 seconds and the season test about 2 minutes (the
planner per method per deadline, four at a time). Results are cached in
`data/processed/decisions/<season>.pkl`, keyed by a fingerprint of everything they
depend on: the replay's predictions and real points, the deadlines, the planner's
dials, and `decisions.VERSION`. So they're only recomputed when a new gameweek
finishes or something changes.

## Tests
`tests/test_decisions.py`:
- FPL's scoring: substitutes, formation limits, the goalkeeper rule, the captain and
  vice;
- weekly picks within budget, with hindsight as the ceiling;
- a season that sells an injured star, with points after hits, bank and free
  transfers following the rules;
- the cache reused until predictions change;
- the tab rendering, with your real team for comparison.
