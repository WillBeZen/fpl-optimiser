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
xp = (1 - w) x points model + w x event model        w = config.EVENT_MODEL_WEIGHT (0.5)
```

The **points model** (gradient-boosted trees, below) predicts points directly; the
**event model** builds them up from football events (team ratings → fixture goals →
each player's goals, assists, clean sheets...). Each is good at something: the trees
learn from everything in the player's record; the event model prices each fixture
explicitly. The backtest shows the blend beats both.

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

## Does it work? The walk-forward backtest
`model/backtest.py`: for each gameweek *r* of 2026/27, fit on **only** what was known
before *r* (2025/26 plus this season's earlier rounds), predict *r*, and compare with
what happened. A test checks that no prediction ever trains on its own gameweek or
later. Baselines anyone could compute: recency-weighted **form**, and the plain
**season average**.

2026/27 GW2–7, 3,941 predictions (finished matches only):

| method | RMSE | rank corr | top-11 points per GW |
|---|---|---|---|
| **Blend (what the planner uses)** | **2.14** | **0.66** | **56.7** |
| Points model | 2.20 | 0.65 | 55.8 |
| Event model | 2.13 | 0.66 | 52.8 |
| Form (recency-weighted) | 2.25 | 0.61 | 39.3 |
| Season average | 2.44 | 0.65 | 44.3 |

The blend weight was chosen here: w = 0.5 gave the best top-11 and nearly the best
RMSE and rank correlation (w from 0 to 1 in steps; RMSE fell steadily towards the
event model, top-11 peaked at 0.5–0.6).

*RMSE* is the typical miss per fixture (lower is better; it is the right score for an
*expected* value). *Rank corr* is how well it orders players each week. *Top-11* is the
real points of the 11 players each method rated highest, which is closest to what
picking a team needs: the blend's picks scored about **17 more points per gameweek**
than form's. The tree settings were chosen by this backtest (three candidates; the
more regularised one won on every score). Caveat: seven gameweeks is a small sample,
and 2025/26 contributes only its top 50 players (see Section 2).

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
