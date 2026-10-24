# 02 — Feature engineering

**Goal:** turn the raw tables from Section 1 into the table the model trains on —
one row per **(player, gameweek)** the player featured in, with predictive
features and the targets we'll learn (`target_points`, `minutes`). Four ideas do
the work: **recency weighting**, **opponent characteristics/archetypes**, a strict
**no-leakage** assembly, and **last season as a prior**. Build it with:

```bash
python scripts/pull_fpl.py          # file the live season (Section 1)
python scripts/build_features.py    # every filed season -> data/processed/training_table.csv
python scripts/build_features.py --seasons 2026-27
```

## 1. Recency weighting (`features/recency.py`)

Recent form should count more than old form. We weight a match played `g`
gameweeks ago by

```
weight = 0.5 ** (g / half_life)
```

`half_life` (default `config.RECENCY_HALF_LIFE_GW = 8`) is the number of gameweeks
at which a match counts **half** as much as the most recent one. Worked example
with half-life 8: last week ≈ 0.92, four weeks ago ≈ 0.71, eight weeks ago = 0.50,
sixteen ago = 0.25. A player's "form xGI" is then a weighted average of his
per-match xGI using those weights — a smooth, recency-biased number rather than a
flat season average. Lower the half-life to chase hot streaks; raise it to trust
the longer record. (Section 3's backtest is how you'll tune it, not guesswork.)

## 2. Opponent characteristics + archetypes (`features/opponent.py`)

Instead of treating the opponent as one of 20 labels, we describe each team by a
small **profile**: goals conceded per game (from the Section 1 adapter) and FPL's
own home/away defensive-strength ratings. We then standardise those and run
**k-means** (`config.ARCHETYPE_K = 3`) to cluster teams into defensive
**archetypes**, relabelled so **archetype 0 = meanest defence** and the highest
index = leakiest. `archetype_summary()` prints which teams fall in each — that's
the human-readable "type" Claude can refer to ("a leaky, archetype-2 defence").

This is the seed of your "Haaland vs a certain type of defence" idea: the model
sees *what kind of defence* a player is facing, not just which club, so it can
generalise to a newly promoted side it has never seen.

**Honest limits of v1.** With the current data we have defensive *quality* (goals
conceded, FPL strength) but not yet defensive *style* (high-press vs low-block),
which needs xGA/PPDA — those are NaN until a richer source is reachable (Section 1
notes why). The moment one is, you add the columns to `PROFILE_FEATURES` and
re-cluster; nothing downstream changes.

**Leakage caveat.** These profile values are season-level/current, so for the
strict 2025/26 backtest they mildly "peek" at full-season quality. The
*quantitative* opponent signal used per row — `opp_gc_pg_asof` — is computed
leakage-free (next section). Archetypes are coarse, season-stable descriptors.

## 3. The training table (`features/build.py`)

`build_training_table(...)` produces one row per (player, gameweek):

| group | columns |
|---|---|
| keys | `id`, `round`, `web_name`, `position`, `was_home`, `opponent_team` |
| targets | `target_points`, `minutes` |
| player form (recency-weighted, **prior matches only**) | `form_points_pg`, `form_minutes_pg`, `form_xg_pg`, `form_xa_pg`, `form_xgi_pg`, `form_defcon_pg`, `form_bonus_pg`, `form_bps_pg`, `form_starts_rate`, `games_played` |
| opponent | `opp_gc_pg_asof` (leakage-free), `opp_archetype`, `opp_str_def` (venue-correct) |
| interaction | `x_xgi_oppgc` = attacking form × opponent leakiness |
| cold-start | `is_cold` |
| last season (section 4) | `has_prev_season`, `prev_points_per_gw`, `prev_minutes_per_gw`, `prev_points_per_90`, `prev_xgi_per_90`, `prev_defcon_per_90`, `prev_changed_team` |
| multi-season table only | `season`, `code` (the player's permanent id), `sample_weight` |

**No leakage — the rule that makes a backtest trustworthy.** Every feature for
round *t* is computed from rounds **strictly before *t***. `weighted_form_asof`
filters the player's history to `round < t`; `team_defence_asof` builds each team's
goals-conceded only from its earlier matches. So a row never contains information
that wasn't available before that gameweek kicked off. (Verified by tests: the
round-3 form for a player ignores his round-3 points entirely, and rewriting
everything that happened in GW3 changes no feature of any GW1–3 row.)

**Fast, but the same maths.** `weighted_form_asof` is the readable definition of
one row's form. The table builder computes identical numbers for every row at
once: per player it builds one small matrix `W[i, j]` = the weight match *j* gets
when predicting match *i* (zero unless *j* was in a strictly earlier round), so
each form feature is a single matrix product. A full season (~32k rows) builds in
under half a second, against 3+ minutes for the old row-by-row loop, with output
identical to it. A test checks the two agree row by row.

**Venue-correct opponent strength.** If the player was at home, his opponent
played *away*, so we attach the opponent's **away** defensive rating (and vice
versa).

**Interaction term.** `x_xgi_oppgc` multiplies a player's attacking form by how
leaky the opponent has been. A tree model can find such interactions on its own,
but handing it this one cheaply makes the "good attacker vs leaky defence" effect
easy to pick up — and easy to explain.

**Cold-start.** Early-season rows (and debutants) have little or no history. We flag
them (`is_cold`, under 3 games this season). A player with a 2025/26 record gets
last season as his prior (section 4). Anyone still without form is filled with
**position-median** form, taken (like everything else) only from rows in
**earlier** rounds: a GW10 debutant gets his position's median form over GW1–9.
Rows with nothing earlier at all stay NaN (LightGBM handles NaN natively).

## 4. Last season as a prior (`features/prior_season.py`)

A 2026/27 row can lean on the same player's 2025/26 record, found through the
cross-season link table (Section 1: matched on FPL's permanent player `code`).
Last season is entirely in the past, so this can't leak.

**Blended into form, through the same recency weighting.** Last season enters each
row's form as if it had been played just before GW1: its 38 gameweeks at the
player's per-gameweek averages, ending `config.PREV_SEASON_GAP_GW` gameweeks
(default 8, standing in for the summer break) before GW1. So at GW1 his form *is*
last season's average, and as this season's (more recent) matches arrive their
weight takes over. With half-life 8 and gap 8, last season counts like ~5 recent
matches at GW1, ~2 by GW10 and under 1 by GW20. Raise the gap to trust last season
less. `weighted_form_asof(..., prior=...)` is the readable definition (it literally
appends the virtual matches); the table builder uses the equivalent closed form, and
a test checks they agree.

**And as explicit context:** `prev_points_per_90`, `prev_xgi_per_90` and
`prev_defcon_per_90` (left NaN under 270 minutes, too noisy), points and minutes
per gameweek, whether he changed club, and `has_prev_season`.

On the real data, 73% of 2026/27 rows have a 2025/26 prior, and GW1 rows went from
no form at all to form for 77% of players. The rest are genuine arrivals (promoted
squads, new signings), who use the position prior instead.

*Approximation:* per-gameweek averages divide season totals by 38, so a January
arrival looks like a part-timer until this season's matches correct it.

## 5. One table across seasons (`features/multi_season.py`)

`build_all()` builds every filed season on its own (ids only mean something within a
season), each with the previous season as its prior when that season is filed, then
stacks them with `season`, `code` and `sample_weight`: 1 for the latest season,
`config.PREV_SEASON_WEIGHT` (0.5) for the one before, its square for two back.
Section 3 passes `sample_weight` to the model so recent seasons count more.

**Coverage caveat:** the 2025/26 per-match rows exist only for the 50 players fetched
before the API moved on, and they are the season's top scorers, a biased sample. The
2025/26 *season totals* cover all 841 players, which is what the prior uses. Full
2025/26 match-by-match data would need a public archive (for example the
`vaastav/Fantasy-Premier-League` repository); the season store can take it as
another source.

## 6. Player types for visual analysis (`features/player_profiles.py`)

The opponent-archetype idea, applied to players, for the HTML report (Section 7).
Each player is described by per-90 rates (xG, xA, defensive contributions, saves,
xG conceded, bonus) plus his **minutes share**. Profiles blend this season with last
season at `PREV_SEASON_WEIGHT` per minute, so early-season profiles aren't 300
minutes of noise. Each position is then clustered (standardised k-means) into
`config.PLAYER_ARCHETYPE_K` types, three per position because a scatter can only
keep three colours distinct for colour-blind readers. Types are ordered by points
per 90 and named by what stands out in their centre, e.g. *High bonus · High xA*
(creators like Cherki and B. Fernandes) or *High def. contributions · Low xG*
(ball-winners). Players need 450+ blended minutes to be profiled (340 of 668 today).
Not a model input yet: they are for looking at.

## Tests
- `tests/test_features.py` — decay halves at the half-life; weighted mean favours
  recent; archetypes order by goals conceded (and survive a feature missing for
  every team); profiles join on reconciled names; `weighted_form_asof` excludes the
  current and future rounds; `team_defence_asof` uses prior rounds only, keys double
  gameweeks on (team, round) and copes before any fixture has finished; the
  vectorised table matches `weighted_form_asof` row by row (with and without a
  prior); venue-correct opponent strength; a whole-table **no-leakage** test with and
  without last season; leakage-free cold-start priors; the prior table from linked
  totals; last season seeding GW1 and fading week by week; an end-to-end smoke test.
- `tests/test_multi_season.py` — two tiny filed seasons through store → link →
  prior → table: stacking, sample weights, the same player under two ids.
- `tests/test_player_profiles.py` — per-90 blending across seasons and types that
  split by style and rank by points.

## What's deferred to Section 3
The minutes model (will he play, and for how long), the points model itself
(LightGBM on this table, recency-weighted samples), and the walk-forward backtest
that tells you whether any of these choices actually help.