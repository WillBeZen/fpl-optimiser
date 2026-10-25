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
| blend weight | chosen at each deadline only from **earlier** gameweeks' out-of-sample predictions (lowest RMSE); config's weight before there is any |
| player pool | the planner's candidate pool **as it would have been**: `optimise/pool.py` on r's own forecast, plus your squad at r (FPL's past picks) |
| injury flags | **not possible:** FPL doesn't archive them, so replayed predictions assume everyone fit. `model/forecast_log.py` keeps the real deadline forecasts from now on |

**The methods**, all scored on exactly the same player-gameweeks:

| method | what it is |
|---|---|
| Blend, weight from earlier weeks | the planner's blend, weight chosen walk-forward: **the reference** |
| Blend, config weight (in-sample) | the blend at `EVENT_MODEL_WEIGHT` (0.5), which was chosen by looking at these weeks: optimistic |
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

**2. Model vs simple methods**, the planner's pool (1,535 player-gameweeks):

| method | RMSE | rank corr. | top 11 / GW |
|---|---|---|---|
| Blend, weight from earlier weeks | 2.784 | 0.574 | 51.7 |
| Event model | 2.766 | 0.576 | 54.0 |
| Position/price × P(plays) | 2.798 | 0.553 | 59.3 |
| Points model | 2.885 | 0.539 | 44.0 |
| Recent points (EWMA) | 2.947 | 0.537 | 41.5 |
| Season average | 3.270 | 0.525 | 45.7 |

- On **accuracy**, the models beat every baseline. Across every player, the blend's
  RMSE is better than each baseline in 6 of 6 gameweeks, with 90% intervals that
  exclude zero.
- On **picking** (top 11), the blend does **not** beat position/price × P(plays): that
  baseline was ahead in 5 of 6 gameweeks (−7.7 ± 4.9 points, interval −17.6 to +2.3).
  Its edge comes from the minutes model; position/price with no model at all is far
  worse at ordering players (rank correlation 0.36 against 0.66 for every player).
- Walk-forward, the blend's weight goes to 1.0 (the event model alone) from GW3. The
  tree-based points model doesn't earn its place in the blend on this evidence.
- Top 11 swings by 10–20 points from week to week, so six gameweeks can't separate
  close methods on it.

**3. The old pool's bias was look-ahead.** The previous Accuracy tab used *today's*
pool for past gameweeks, which favours players who did well since: bias −0.34. With
pools rebuilt at each deadline it's +0.13.

**4. Half-life:** 4, 8 and 16 gameweeks score the same within noise. It isn't tuned
on these weeks (that would be circular); it's chosen on last season and confirmed on
this one.

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
| Blend weights chosen from earlier weeks only | each deadline's weight used only earlier gameweeks |
| Predictions are the planner's blend | `(1 − w) × points model + w × event model`, none missing |
| Predictions use the half-life in config.py | recorded on every prediction |
| Pools rebuilt at each deadline | each from that deadline's own forecast |
| Every prediction has a match record | every predicted player-fixture is in FPL's records |
| Totals add up | the same whether added by player, by gameweek or match by match |

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

## Cost
About 2 minutes on a full run:
- the replay: one model fit per deadline, about 7 seconds each;
- the half-life replays: one gameweek per deadline;
- the code-path check: one more fit.

`--quick` skips it.

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
