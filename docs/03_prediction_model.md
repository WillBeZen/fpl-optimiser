# 03 — Prediction model

**Goal:** expected FPL points for every player in each of the next few gameweeks,
which is what the optimiser (Sections 4–5) maximises, plus evidence that the
numbers beat simple rules of thumb.

```python
from fpl_optimiser.model import predict
fc = predict.forecast()          # fits on every filed season, predicts the next N GWs
fc.wide()                        # one row per player: GW9 ... and the total
fc.per_fixture                   # one row per player x fixture, with the event breakdown
```

The expected points the optimiser uses blend two models:

```
xp = (1 - w) x points model + w x event model        w = config.EVENT_MODEL_WEIGHT (0.70)
```

The **points model** (gradient-boosted trees, below) predicts points directly; the
**event model** builds them up from football events (team ratings → fixture goals →
each player's goals, assists, clean sheets...). Each is good at something: the trees
learn from everything in the player's record; the event model prices each fixture
explicitly. The weight (0.70) and the recency half-life (16 gameweeks) are chosen on
last season by rules fixed in advance, then checked on this season
(`scripts/study_last_season.py`, `docs/08_accuracy.md`). On last season's 37 gameweeks
any weight from 0.6 to 0.75 predicts almost equally well. The blend is more accurate
than the points model alone, level with the event model, and better than every simple
method.

## The points model: will he play, and what if he does?

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
60-minute rate), recent minutes (the latest match's minutes, 60+ minutes in the last
three, matches missed in a row), the opponent's recency-weighted goals conceded (and
its product with attacking form), venue, the last-season context columns, an
`earlier_season` flag, and position. Training uses the multi-season table (Section 2) with its `sample_weight`, so
2025/26 rows count half.

**Every input must be knowable at the deadline.** Two columns the table carries for
the report are not inputs:
- the opponent's "style", clustered on goals conceded over the whole season so far.
  For a past match that includes the match itself, which leaked the result into the
  backtest; it inflated the points model's top-11 by about 14 points a gameweek;
- FPL's defence rating, a current snapshot, 0 for every club this season. Its only
  signal was which season a row came from, which `earlier_season` now says explicitly.

Inputs are rounded to 6 decimals (`inputs.ROUND_TO`), so the training table and the
forecast, which compute the same features with different arithmetic, can never land a
value on different sides of a tree split.

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

## The event model (`model/events.py`)
A transparent football model, everything computed as of a round from data strictly
before it (so the same code serves the forecast and the backtest):

1. **Team ratings.** Attack = recency-weighted xG created per game; defence = xG
   conceded per game (team xG is the sum of its players' xG, used only when their
   records cover a whole team, 900+ minutes; otherwise goals). Last season's goals
   for/against per game blend in as a fading prior with the same half-life and summer
   gap as player form; promoted clubs start from the relegated clubs' record.
2. **Fixture goals.** home goals = league average × home advantage × home attack index
   × away defence index (mirror for the away side). Clean sheet = exp(−opponent's
   expected goals).
3. **Player events.** His recency-weighted per-90 rates (xG, xA, saves, bonus, cards,
   how often he reaches the defensive-contribution threshold), shrunk towards his
   position's average when he has few minutes (270 minutes' / 3 matches' worth),
   scaled by the fixture (this fixture's team goals vs his team's usual) and by his
   expected minutes from the **minutes model** (P(plays), P(60+)), so both models agree
   on who plays.

Points by source then follow FPL's scoring table in `config.SCORING`: appearance,
goals, assists, clean sheet, goals conceded (−1 per 2, Poisson), saves (1 per 3),
defensive contributions, bonus and cards. `xp_events` is their sum.

## Availability (`predict.availability`)
Next gameweek, `xp` and `p_play` are scaled by FPL's `chance_of_playing_next_round`
(75% → ×0.75; injured or suspended with no chance given → 0). Later gameweeks follow
the news: "Expected back 8 Nov" / "Suspended until 31 Oct" keep him out until the
first gameweek after that date; "Unknown return date" brings him back gradually at
`config.UNKNOWN_RETURN_RAMP` (25%, 50%, 75%, then fit), so the planner neither writes
him off nor counts on him straight away. Players who have left (`u`) or aren't in the
squad (`n`) are 0 throughout. `xp_raw` keeps the unscaled value.

## Does it work? The honest replay
`model/evaluation.py` replays every finished gameweek as it stood at its deadline
(models, features, players, prices and the planner's pool from before it only). It
scores the blend against the points and event models and four simple baselines on the
same player-gameweeks, with gameweek-by-gameweek uncertainty. Full method and results:
**`docs/08_accuracy.md`**.

In short (2026/27 GW2–7):
- The models beat every simple baseline on **accuracy** (RMSE).
- On **picking** (top 11), the blend doesn't beat a position/price baseline that uses
  the minutes model's P(plays).
- Walk-forward, the event model takes the whole blend.

The earlier backtest here reported better numbers for the points model: an input
leaked results (see Inputs above), and the blend weight and tree settings were chosen
on the same weeks they were judged on. `model/backtest.py` (`walk_forward`) remains as
the training-table path that the replay is checked against.

## Tests
`tests/test_model.py`: the models learn an obvious signal (regulars play and score,
fringe players don't); `p_60 ≤ p_play` and `xp = p_play × pts_if_play`; one-class and
tiny data fall back gracefully; the backtest never trains on the future; availability
flags and return dates from the news; a full forecast from a filed store (fixture text,
blank gameweeks as 0) and the blend. `tests/test_events.py`: team ratings rank attack
and defence and ignore the future, last season fades and promoted clubs start low,
team xG needs a whole team's records, fixture goals and clean sheets, the Poisson
helpers, shrunk per-90 rates, and points by source adding up. The train/predict
consistency test lives in `tests/test_features.py`.

## Explainers (`model/explain.py`)
For the report's "The model" tab, none of which changes a prediction: the recency
weight curve and last season's share by gameweek, one player's form story (each
match's weight), form under other half-lives, permutation importance for both tree
models, how the points model responds to the opponent's defence, each player's
expected points against every club, what each player type has scored against each
defensive style, and who FPL's flags cut. Tested in `tests/test_explain.py`.
