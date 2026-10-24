# 04 — Optimisation: the best squad for one gameweek

**Goal:** given everyone's expected points (Section 3), pick the 15-man squad, the
starting XI, the captain and the vice-captain that score the most, without breaking
a single rule.

```python
from fpl_optimiser.optimise import squad
best = squad.pick_squad(forecast.xp[forecast.xp.gw == forecast.next_gw])
best.players, best.formation, best.captain, best.xp, best.cost
```

## Why a mixed-integer program
Every player is a yes/no decision: in the squad? starting? captain? The rules are
straight-line (linear) limits on those decisions, so a solver can search every legal
combination and return the provably best one, not just one that looks good. It uses
PuLP with the CBC solver bundled inside PuLP 3, so nothing else needs installing.

**The rules** (all from `config.py`): exactly 15 players (2 GK, 5 DEF, 5 MID, 3 FWD);
total price within the budget; at most 3 per club; an XI of 11 in a legal formation
(1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD); one captain and one vice-captain, both starting.

**The objective:** the XI's expected points, plus the captain's again (he scores
double), plus the bench at `config.BENCH_WEIGHT` (0.1). Bench players only score when a
starter doesn't play, so they're worth something, but not much. A tiny tie-break
makes the best non-captain starter the vice-captain.

**The team sheet** lists the XI by position and then the bench in FPL's order: the
reserve keeper first, then outfielders by expected points (the order of automatic
substitutions).

`must_include` / `exclude` force players in or out (e.g. "keep Haaland", "not him").
An impossible request (too small a budget, too many exclusions) raises a clear error
instead of returning a broken squad.

## Example (real data, GW9 2026/27)
A from-scratch squad (i.e. a wildcard): 3-4-3, £99.9m, 56.1 expected points with
B. Fernandes captain. It picks three Fulham players, which the club limit stops at
exactly three.

## Tests
`tests/test_optimise.py`: every rule holds (quotas, budget, club limit, formation,
captain and vice in the XI, captain = the best starter); more budget never scores
less; forced picks and exclusions are honoured; the club limit binds when one club's
players are all the best; an impossible budget is reported.
