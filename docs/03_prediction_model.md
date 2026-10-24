# 03 — Prediction model

**Goal:** expected FPL points for every player in each of the next few gameweeks,
which is what the optimiser (Sections 4–5) maximises, plus evidence that the
numbers beat simple rules of thumb.

```python
from fpl_optimiser.model import predict
fc = predict.forecast()          # fits on every filed season, predicts the next 5 GWs
fc.wide()                        # one row per player: GW9 ... GW13 and the total
```

## Two parts: will he play, and what if he does?

```
xp = p_play × pts_if_play
```

- **Minutes model** (`model/minutes.py`): two classifiers, `p_play` = P(minutes > 0) and
  `p_60` = P(60+ minutes). Minutes are the biggest single driver of points.
- **Points model** (`model/points.py`): a regressor trained **only on matches the player
  played**, so it learns what he scores *when he plays*.

Splitting it this way means injury news scales exactly the part it should (see
availability below), and neither model has to learn "points" and "will he be picked"
as one blurred thing.

## Inputs and training
One feature list (`model/inputs.py`) feeds both models: price at the time, recency-
weighted form (points, minutes, xG, xA, xGI, defensive contributions, bonus, BPS,
60-minute rate), opponent leakiness and archetype, venue, the last-season context
columns, and position. Training uses the multi-season table (Section 2) with its
`sample_weight`, so 2025/26 rows count half.

**Gradient-boosted trees, via scikit-learn's HistGradientBoosting.** It is the LightGBM
algorithm with native missing-value handling. LightGBM itself needs an extra system
library on macOS (`libomp`); this needs nothing beyond `requirements.txt`. The trees
are deliberately small and regularised (8 leaves, at least 80 rows per leaf), because
there are only a few thousand training rows. Two safeguards: an input with no values
at all becomes a constant (the trees ignore it), and below 160 rows the models fall
back to plain averages rather than fitting noise.

## Predicting the next gameweeks
`features/build.build_prediction_table` builds one row per (player, upcoming fixture),
with every feature computed **as of the next deadline**, by exactly the code that builds
the training table. A test checks that a prediction made before a gameweek equals the
training row that gameweek later gets, feature for feature. A double gameweek gives a
player two rows, which are added up; a blank gameweek scores 0.

**Availability** (`predict.availability`): next gameweek, `xp` and `p_play` are scaled
by FPL's `chance_of_playing_next_round` (75% → ×0.75; injured or suspended with no
chance given → 0). Later gameweeks assume he's back, because FPL publishes nothing;
the Claude advisor reads the news for longer absences. Players who have left (`u`) or
aren't in the squad (`n`) are 0 throughout. `xp_raw` keeps the unscaled value.

## Does it work? The walk-forward backtest
`model/backtest.py`: for each gameweek *r* of 2026/27, fit on **only** what was known
before *r* (2025/26 plus this season's earlier rounds), predict *r*, and compare with
what happened. A test checks that no prediction ever trains on its own gameweek or
later. Baselines anyone could compute: recency-weighted **form**, and the plain
**season average**.

2026/27 GW2–8, 4,609 predictions:

| method | RMSE | rank corr | top-11 points per GW |
|---|---|---|---|
| **Model** | **2.19** | **0.61** | **49.6** |
| Form (recency-weighted) | 2.22 | 0.58 | 38.9 |
| Season average | 2.40 | 0.62 | 40.6 |

*RMSE* is the typical miss per fixture (lower is better; it is the right score for an
*expected* value). *Rank corr* is how well it orders players each week. *Top-11* is the
real points of the 11 players each method rated highest, which is closest to what
picking a team needs: the model's picks scored about **11 more points per gameweek**
than form's. The tree settings were chosen by this backtest (three candidates; the
more regularised one won on every score). Caveat: seven gameweeks is a small sample,
and 2025/26 contributes only its top 50 players (see Section 2).

## Tests
`tests/test_model.py`: the models learn an obvious signal (regulars play and score,
fringe players don't); `p_60 ≤ p_play` and `xp = p_play × pts_if_play`; one-class and
tiny data fall back gracefully; the backtest never trains on the future; availability
flags; and a full forecast from a filed store (fixture text, blank gameweeks as 0). The
train/predict consistency test lives in `tests/test_features.py`.
