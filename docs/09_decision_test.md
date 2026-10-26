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

## This season (2026/27 GW2–7)
At the current settings (half-life 16, weight 0.70, last season's full match data, the
recent-minutes inputs):

| method | weekly team, pts / GW | season with transfers, pts after hits |
|---|---|---|
| Blend (what the planner uses) | 65.7 | 350 |
| Event model | 63.8 | 396 |
| Position/price × P(plays) | 60.2 | 359 |
| Points model | 59.3 | 311 |
| Season average | 51.8 | 284 |
| Recent points (EWMA) | 51.3 | 301 |
| Hindsight best | 155.0 | — |

Over six gameweeks only one difference is clear: the blend's weekly teams beat recent
points' (+14.3, interval +3.7 to +24.9). Against position/price × P(plays) the blend is
5.5 points a gameweek ahead, but the interval runs from −7.2 to +18.2. Your real team
scored 323 over the same gameweeks. In the season with transfers the blend's squad did
30 points worse than keeping its first squad: one path through six weeks, which is why
the weekly test is the one to read. Last season's 37 gameweeks give firmer answers
(below).

(Before last season's full data, at half-life 8 and weight 0.5, the points model's
weekly teams scored 48.5 a gameweek: the archive mattered most to it.)

## Last season (2025/26 GW2–38, 37 gameweeks)
The same two tests on last season, from the study that chooses the settings
(`scripts/study_last_season.py`: half-life 16, the planner's blend at weight 0.70,
with the recent-minutes inputs). The dashboard shows them under "Last season".

| method | weekly team, pts / GW | blend minus method, 90% interval | season with transfers, pts after hits |
|---|---|---|---|
| Blend (what the planner uses) | 62.8 | | 2,061 |
| Event model | 63.1 | −0.2 (−2.2 to +1.7) | 2,111 |
| Points model | 61.5 | +1.3 (−3.0 to +5.6) | 2,147 |
| Recent points (EWMA) | 58.2 | +4.6 (+0.5 to +8.7) | 1,975 |
| Season average | 57.9 | +4.9 (+0.9 to +8.9) | 2,028 |
| Position/price × P(plays) | 54.3 | +8.5 (+3.6 to +13.4) | 1,926 |
| Hindsight best | 154.6 | | — |

With 37 gameweeks the weekly test separates the methods. The blend's teams beat
position/price × P(plays) by about 8.5 points a gameweek, where this season's six weeks
can't tell them apart, and they beat recent points and the season average too. They are
level with each model alone. In the season with transfers the points model finished
first (2,147), then the event model (2,111) and the blend (2,061). That is one path
through a season, so the order among them is illustrative, but all three end 100–200
points ahead of position/price × P(plays). Every method gained from transferring,
between 273 and 555 points over keeping its first squad.

A squad kept all season can end up with four players from one club when one of them
moves (27 players changed Premier League club during 2025/26). Picking the best XI of a
kept squad therefore lifts the club limit; the planner's own transfers still respect it.

## Tests
`tests/test_decisions.py`:
- FPL's scoring: substitutes, formation limits, the goalkeeper rule, the captain and
  vice;
- weekly picks within budget, with hindsight as the ceiling;
- a season that sells an injured star, with points after hits, bank and free
  transfers following the rules;
- the cache reused until predictions change;
- a kept squad still getting its XI after a player moves club;
- the tab rendering, with your real team for comparison.
