# 08 — Accuracy: predicted vs actual

**Goal:** see how the model's predicted points lined up with what players actually
scored, honestly and checkably, and whether the recency half-life is set right.

```bash
python scripts/run_week.py --open          # the dashboard's "Accuracy" tab
```

## How it works (`fpl_optimiser/model/accuracy.py`)

**The window.** The last `ACCURACY_WINDOW_GW` (5) gameweeks FPL has marked finished.
A gameweek still being played is left out until it's finished.

**Predicted = a replay.** Each gameweek in the window is predicted the way the model
would have predicted it at that deadline (`backtest.walk_forward`):
- the points model is refitted on every gameweek **strictly before** it;
- every recency-weighted feature, and the event model's team and player rates, use
  `RECENCY_HALF_LIFE_GW` from `config.py`;
- the prediction is the planner's blend, `(1 − w) × points model + w × event model`.

`forecast_path_check` proves the replay *is* the forecast: it rebuilds the latest
gameweek through the live forecasting code (`build_prediction_table`, the same fitted
models, the event model, the blend). With that week's prices it matches every player
to within 1e-6 points (exactly, in practice).

One thing the replay can't reproduce: FPL's injury flags as they stood at past
deadlines, because FPL doesn't archive them. So from now on every weekly run also
files its forecast for the next gameweek, flags applied (`model/forecast_log.py`,
`data/processed/forecast_log/<season>/gw<NN>.csv`). Once those gameweeks are played,
the tab compares the forecasts actually made as well.

**Actual = FPL's points.** These come from the filed match records, and are checked
against FPL's official gameweek totals: `event/<gw>/live`, filed once per finished
gameweek by `season_store.pull_live` as `event-live_<gw>.json`.

**The players.** The planner's candidate pool: everyone not dominated
(`optimise/pool.py`, about 240), plus your squad.

## What the tab shows
1. **Summary:** predicted and actual totals, bias, typical miss, how many predictions
   were within 2 points, rank correlation, and how much of the best possible top 11
   the model's top 11 captured. Shown for the pool and for every player: the pool is
   chosen for being good now, which favours players who've been doing well, so it is
   under-predicted a little while every player together is nearly unbiased.
2. **Gameweek by gameweek:** chart and table. Each gameweek also gets its half-life
   weight, plus a recency-weighted typical miss and bias.
3. **Every player in the pool:** each gameweek's points scored, with the prediction
   under it, shaded blue (scored more) / grey (about right) / red (scored less); totals,
   difference, typical miss, minutes. Filter by position or club, and sort any column.
4. **Player by player:** a scatter of each player's predicted against actual total.
5. **Calibration:** player-gameweeks grouped by prediction, average predicted against
   average scored.
6. **By position, and by scoring source:** the event model's predicted appearance,
   goals, assists, clean sheets... against FPL's own breakdown of the points.
7. **Biggest misses:** with a reason (didn't play / under 60 minutes / few returns /
   haul), plus how much of the total miss came from players who didn't play at all.
8. **Half-life check:** the same window replayed at half and double the half-life,
   scored on the same players. One window is a small sample: watch it for a few weeks
   before changing `RECENCY_HALF_LIFE_GW`.
9. **As forecast at the deadline:** the logged forecasts, once there are some.
10. **Reconciliation:** every check, also printed in the terminal by `run_week.py`.

## Reconciliation (`accuracy.reconcile`)
| check | what it proves |
|---|---|
| The window is complete | every gameweek in it is finished and checked by FPL |
| Actual points match FPL's official gameweek totals | match records = `event/<gw>/live`, points and minutes, every pool player-gameweek |
| Every finished gameweek this season matches FPL | the same, over the whole season so far |
| FPL's own breakdown adds up | each player's points by source sum to FPL's total |
| Match records are up to date | *info:* FPL's season totals also count matches in play; the records catch up at the next refresh |
| No prediction used later data | each prediction's model was fitted only on earlier gameweeks |
| The replay is exactly what the forecast would have said | the latest gameweek rebuilt through the live forecasting code |
| Predictions use the half-life in config.py | the half-life recorded on the training table and the event-model replay |
| Predictions are the planner's blend | `xp_blend = (1 − w) × xp + w × xp_events` for every prediction, none missing |
| Players are the planner's candidate pool | same count as the planner's pool |
| Totals add up | the same totals whether added by player, by gameweek or match by match |

## First results (GW3–7, 240 players)
Predicted 2,920 points; actual 3,328. Bias −0.34 per player-gameweek (every player:
+0.10); typical miss 1.93; rank correlation 0.56 (every player: 0.68). By source, the
event model under-predicts **assists** most (171 predicted against 294 scored), because
FPL awards assists more generously than xA counts. It also under-predicts **clean
sheets** for the pool's (mostly strong) defences.

## Cost
About 1½ minutes on a full run: the replay itself is the backtest's, reused; the two
extra half-lives rebuild the table and replay the window; the forecast-path check
refits once. `--quick` skips it (the tab then says so).

## Tests
`tests/test_accuracy.py`:
- the window;
- player-gameweeks: doubles added up, missing predictions flagged;
- every summary number, the half-life weights;
- calibration, position, source and misses;
- reconciliation passing when everything agrees, and catching wrong actuals, leakage,
  the wrong half-life, a broken blend and the wrong pool;
- the forecast log;
- on a filed store: the half-life check, and the replay matching the forecast path
  exactly.

`tests/test_report.py` builds the tab from a filed store, with no failing checks.
`tests/test_season_store.py` and `tests/test_ingest.py` cover filing and parsing FPL's
gameweek totals.
