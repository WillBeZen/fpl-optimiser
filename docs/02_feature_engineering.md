# 02 — Feature engineering

**Goal:** turn the raw tables from Section 1 into the table the model trains on —
one row per **(player, gameweek)** the player featured in, with predictive
features and the targets we'll learn (`target_points`, `minutes`). Three ideas do
the work: **recency weighting**, **opponent characteristics/archetypes**, and a
strict **no-leakage** assembly. Build it with:

```bash
python scripts/build_features.py --limit 50   # quick test
python scripts/build_features.py              # full season -> data/processed/training_table.csv
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

**Cold-start.** Early-season rows (and debutants) have little or no prior history,
so their form would be NaN. We flag them (`is_cold`) and fill the gaps with
**position-median** form as a prior, taken (like everything else) only from rows in
**earlier** rounds: a GW10 debutant gets his position's median form over GW1–9.
GW1 rows have nothing earlier to learn from, so they stay NaN (LightGBM handles
NaN natively); Section 3 can fill them from last season.

## Tests
`tests/test_features.py`: decay halves at the half-life; weighted mean favours
recent; archetypes order by goals conceded (and survive a feature missing for every
team); profiles join on reconciled team names; `weighted_form_asof` excludes the
current and future rounds; `team_defence_asof` uses prior rounds only, keys double
gameweeks on (team, round) and copes before any fixture has finished; the
vectorised table matches `weighted_form_asof` row by row; venue-correct opponent
strength; a whole-table **no-leakage** test; leakage-free cold-start priors; and an
end-to-end smoke test of the assembled table.

## What's deferred to Section 3
The minutes model (will he play, and for how long), the points model itself
(LightGBM on this table, recency-weighted samples), and the walk-forward backtest
that tells you whether any of these choices actually help.