# 08 — Accuracy: predicted vs actual, measured honestly

**Goal:** know whether the model's predictions are any good, and whether its
complexity buys anything over simple methods. Every number on the Accuracy tab comes
from replaying past gameweeks **exactly as they stood at their deadlines**.

```bash
python scripts/run_week.py --open          # the dashboard's "Accuracy" tab
```

## The harness (`fpl_optimiser/model/evaluation.py`)
Every finished gameweek *r*, from the second on, is rebuilt as of its deadline:

| piece | as of r's deadline |
|---|---|
| models | refitted on rows strictly before r: earlier seasons, and this season's earlier gameweeks |
| features | every input computed from matches before r, through the **same code as the live forecast** (`predict.forecast_rows`) |
| players | only those in the game at r, at their club and price at r (from their match records) |
| fixtures | r and the rest of the planning horizon, with later results hidden |
| blend | the planner's: `EVENT_MODEL_WEIGHT`, which is chosen on last season (below), so it is out of sample here |
| player pool | the planner's candidate pool **as it would have been**: `optimise/pool.py` on r's own forecast, plus your squad at r (FPL's past picks) |
| injury flags | **not possible:** FPL doesn't archive them, so replayed predictions assume everyone fit. `model/forecast_log.py` keeps the real deadline forecasts from now on |

**The methods**, all scored on exactly the same player-gameweeks:

| method | what it is |
|---|---|
| Blend (what the planner uses) | the planner's blend, at `EVENT_MODEL_WEIGHT` (0.65): **the reference** |
| Blend, weight from earlier weeks | the weight re-chosen at each deadline from **earlier** gameweeks' out-of-sample predictions only (lowest RMSE); config's weight before there are any |
| Points model / Event model | each half of the blend alone |
| Position/price × P(plays) | similar players' (same position, price band) average points per appearance × the minutes model's P(plays) |
| Position/price | similar players' average points per match: no model at all |
| Recent points (EWMA) | recency-weighted points per match (the half-life) |
| Season average | the player's average points per match so far |

**Populations:** the planner's pool at each deadline (what decisions draw from), and
every player.

**Uncertainty:** players aren't independent evidence, because hundreds share each
gameweek's fixtures. The gameweeks are. So each method is compared with the reference
gameweek by gameweek, paired. The tab shows the mean difference, its standard error,
a 90% interval (t-distribution) and how many gameweeks the reference won. The verdict
is "Blend better" or "This method better" only when the interval excludes zero;
otherwise "No clear difference".

## What the honest replay found (2026/27 GW2–7, 6 gameweeks)
**1. A leak in the old backtest.** The points model's "opponent style" input was
clustered on each opponent's goals conceded over the *season so far*. For a past match,
that includes the match itself. Removing it changed the points model's top-11 from 55.8
to 42.2 points a gameweek, while its RMSE barely moved (2.20 to 2.22). The leak had been
flattering its *picks*.

FPL's opponent defence rating (0 for every club this season) only signalled which
season a row came from. It's replaced by an explicit `earlier_season` flag. Every
model input is now knowable at the deadline (`model/inputs.py`).

**2. Model vs simple methods.** At first (before last season's full data, at half-life
8 and weight 0.5) the blend's top 11 trailed position/price × P(plays) by 7.7 points a
gameweek in the planner's pool, and the points model's top 11 was 42.8 across every
player. That prompted the rest of this page. With every player's 2025/26 match records
in the training data (`docs/01_data_ingestion.md`), the points model's top 11 rose to
51.8 and its RMSE fell from 2.226 to 2.149 (every player, same settings). GW2, when
last season is all there is to go on, improved most.

At the current settings (half-life 4, weight 0.65, both chosen on last season), the
planner's pool (1,428 player-gameweeks):

| method | RMSE | rank corr. | top 11 / GW |
|---|---|---|---|
| Blend (what the planner uses) | 2.776 | 0.595 | 52.8 |
| Blend, weight from earlier weeks | 2.785 | 0.596 | 47.3 |
| Points model | 2.786 | 0.595 | 59.7 |
| Event model | 2.798 | 0.574 | 53.5 |
| Position/price × P(plays) | 2.829 | 0.585 | 60.7 |
| Recent points (EWMA) | 3.069 | 0.535 | 46.2 |
| Season average | 3.365 | 0.522 | 44.3 |

- On **accuracy** the planner's blend is the best method. Across every player its RMSE
  (2.097) beats position/price × P(plays) in 5 of 6 gameweeks (interval −0.070 to
  −0.008) and the other baselines in all 6.
- On the **raw top 11**, position/price × P(plays) is ahead (−7.8, interval −15.3 to
  −0.4). But the top 11 ignores price: that baseline rates players by price band, so its
  top 11 is simply the most expensive regulars. Within a £100m squad, which is the
  actual decision, the blend's teams score more (66.0 against 60.2 a gameweek, too few
  gameweeks to call: `docs/09_decision_test.md`). Over last season's 37 gameweeks the
  blend was ahead on both measures (below).
- Re-choosing the weight each week from this season's earlier weeks alone is unstable
  while there are few of them (it went 0.65, 0.05, 0.45, 0.5, 0.45, 0.35): that blend's
  top 11 trails the planner's fixed weight by 5.5 (interval 0.8 to 10.2).
- Top 11 swings by 10–20 points from week to week, so six gameweeks can't separate
  close methods on it.

**3. The old pool's bias was look-ahead.** The previous Accuracy tab used *today's*
pool for past gameweeks, which favours players who did well since: bias −0.34. With
pools rebuilt at each deadline it's +0.13.

**4. Half-life:** on these six gameweeks 2, 4 and 8 score the same within noise
(RMSE 2.782, 2.776, 2.785). The half-life isn't tuned on them (that would be circular):
it's chosen on last season and checked on this one, below.

## Last season: 37 gameweeks, and where the settings come from (`model/tuning.py`)
Six gameweeks can't separate close methods, and tuning a setting on the weeks then
quoted as evidence is circular. So last season (2025/26, with every player's match
records: `docs/01_data_ingestion.md`) is replayed exactly like this one, deadline by
deadline: `python scripts/study_last_season.py`, about 25 minutes, once a season. The
weekly run only reads its saved file.

**The methods over 37 gameweeks** (2025/26 GW2–38 at half-life 4, every player:
28,648 player-gameweeks):

| method | RMSE | rank corr. | top 11 / GW |
|---|---|---|---|
| Blend (what the planner uses: 0.65) | 1.964 | 0.694 | 52.5 |
| Blend, weight from earlier weeks | 1.967 | 0.694 | 52.6 |
| Event model | 1.968 | 0.692 | 52.8 |
| Points model | 1.979 | 0.694 | 51.4 |
| Position/price × P(plays) | 2.000 | 0.684 | 46.6 |
| Recent points (EWMA) | 2.054 | 0.715 | 46.2 |
| Season average | 2.074 | 0.692 | 49.2 |

(The planner's weight was chosen on these weeks, so its row is in-sample here; the
blend whose weight is chosen from earlier weeks is not, and scores the same.)

- The blend is clearly more accurate than every simple method. Its top 11 beats
  position/price × P(plays) by 5.9 points a gameweek (90% interval 1.6 to 10.1, the
  planner's pool, ahead in 23 of 37 gameweeks). This season's first six weeks hinted
  the opposite (−7.7 ± 4.9); with 37 the answer is clear.
- Recent points rank *every* player better (0.715, ahead in 36 of 37 gameweeks), mostly
  by separating who plays from who doesn't. Within the planner's pool the two rank
  equally, and recent points' top 11 is 6 points a gameweek worse.
- The blend is more accurate than either model alone: lower RMSE than the points model
  in 33 of 37 gameweeks, and slightly lower than the event model's. At picking, the
  three are within noise.
- The decision test agrees (`docs/09_decision_test.md`): the blend's weekly teams beat
  every simple method's.

**The half-life.** The rule, fixed before looking: the lowest blend RMSE over every
player-gameweek, the current setting kept unless beaten by more than 0.002.

| half-life | blend RMSE | rank corr. | top 11 / GW |
|---|---|---|---|
| **4 (chosen)** | **1.9661** | 0.694 | 52.5 |
| 6 | 1.9754 | 0.688 | 51.8 |
| 8 (before) | 1.9814 | 0.684 | 52.3 |
| 12 | 1.9868 | 0.680 | 51.2 |
| 16 | 1.9900 | 0.677 | 54.6 |

Shorter is more accurate at every step. Against 8, a half-life of 4 has lower RMSE in
30 of 37 gameweeks (−0.015, interval −0.019 to −0.011) and ranks better in 36, with no
difference in picking (+0.3, interval −1.9 to +2.5). A caveat, reported rather than acted
on: 16 picked a better top 11 than 8 (+2.4, interval +0.9 to +3.8) while being the least
accurate. With five half-lives compared, one result like that can be chance, which is
why the rule was fixed first.

**The weight**, at half-life 4: 0.65 had the lowest RMSE, but the curve is flat (0.5 to
0.75 are within 0.001). The weight chosen from earlier weeks settled at 0.65 in 27 of 37
gameweeks.

**Checked on this season** (2026/27 GW2–7, every player, 3,941 player-gameweeks):
half-life 4 and weight 0.65 against the settings before (8 and 0.5):
- RMSE 2.097 against 2.125: lower in 5 of 6 gameweeks (−0.028, interval −0.050 to
  −0.007);
- top 11: 52.8 against 54.2, no clear difference (−1.3, interval −8.9 to +6.2).

Six gameweeks can't prove the choice better; the check is there to catch one that does
clearly worse, and it doesn't.

**What last season can't test:** it had no season before it filed, so none of its
players had a prior. How fast last season's prior fades early in a season (which the
half-life also sets) only shows in this season's early gameweeks, which the check above
covers.

## Data checks (`accuracy.reconcile`): about the data, not a score
| check | what it proves |
|---|---|
| Every replayed gameweek is finished | finished and checked by FPL |
| Actual points match FPL's official gameweek totals | the match records equal `event/<gw>/live`, points and minutes |
| Every finished gameweek this season matches FPL | the same, over the whole season |
| FPL's own breakdown adds up | each player's points by source sum to his total |
| Match records are up to date | the records add up to FPL's season totals. `pull_live` now re-fetches any player whose club has played since his file was saved, so a mid-gameweek refresh no longer leaves them behind |
| No prediction used later data | each deadline's models, features, baselines and pool came only from earlier gameweeks |
| Two code paths give the same predictions | the latest gameweek predicted from the training table (`backtest.walk_forward`) and from the forecasting code agree to 1e-9, player by fixture |
| Blend weights chosen from earlier weeks only | each deadline's weight for "weight from earlier weeks" used only earlier gameweeks |
| Predictions are the planner's blend | `(1 − w) × points model + w × event model` at config.py's weight, none missing |
| Predictions use the half-life in config.py | recorded on every prediction |
| Pools rebuilt at each deadline | each from that deadline's own forecast |
| Every prediction has a match record | every predicted player-fixture is in FPL's records |
| Totals add up | the same whether added by player, by gameweek or match by match |
| Last season's archive matches the fixture list | every archive record's gameweek, opponent, venue and score agree with FPL's fixtures |
| Last season's archive agrees with FPL's own match files | the 50 players FPL's files cover: identical records |
| Last season's archive adds up to FPL's season totals | every player, 18 stats (after dropping the archive's repeated rows) |

*Why "two code paths" needed a fix.* Every numeric input is now rounded to 6 decimals
(`inputs.ROUND_TO`). The two paths compute the same features with different arithmetic
(~1e-16 apart), and a value sitting exactly on a tree split could fall either side:
one GW5 prediction moved by 0.085.

## The rest of the tab
- **Every player, the last `ACCURACY_WINDOW_GW` gameweeks:** points scored with the
  prediction under them, shaded by over/under-prediction, for weeks in that deadline's
  pool ("not in pool" otherwise).
- **Player totals chart, calibration, by position and by scoring source** (the event
  model's sources against FPL's own breakdown), **biggest misses**, the **half-life
  check**, and **as forecast at the deadline** (logged forecasts, from GW9).
- **Last season:** the study below, read from its saved file: the methods over 37
  gameweeks, the half-life table, the weight curve, and the check on this season.

## Cost
About 2 minutes on a full run:
- the replay: one model fit per deadline, about 7 seconds each;
- the half-life replays: one gameweek per deadline;
- the code-path check: one more fit.

`--quick` skips it. The last-season study (`python scripts/study_last_season.py`) takes
about 25 minutes and is run once a season, or after a model change: the weekly run only
reads its saved file (`data/processed/last_season/<season>.pkl`).

## Tests
`tests/test_evaluation.py`:
- players, prices and clubs as they stood;
- later results hidden; training stopping before the deadline;
- the walk-forward weight; the baselines from training rows only;
- the scoring and paired-difference maths;
- a full replay on a filed store, including the half-life guard;
- every model input knowable at the deadline.

`tests/test_accuracy.py`: the data checks passing when everything agrees, and
catching:
- wrong actuals;
- leakage;
- the wrong half-life;
- a broken blend;
- a weight chosen from later weeks;
- missing records;
- missing pools;

plus the half-life check and the two-code-path check on a filed store.

`tests/test_tuning.py`: the half-life rule (the current setting stays unless clearly
beaten), the weight curve, two blends compared gameweek by gameweek, the study's order
(quick replays, the choice, then one full replay), filing it, and the tab sections.

`tests/test_archive.py`: git's blob id, repeated rows dropped, the archive filling in
only the players FPL's files don't cover (FPL's record wins), a test store never reading
the real archive, the checks catching wrong points, a missing match and a wrong venue,
and a download kept only if it is the pinned file.
