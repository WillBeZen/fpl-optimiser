# Project charter — FPL optimiser

## What this builds

A weekly Fantasy Premier League decision system with two **separable** engines:

1. **Prediction engine (ML / statistics).** Estimates each player's *expected FPL
   points* for the coming gameweek(s). Recency weighting and the
   "characteristics, not teams" opponent model live here.
2. **Optimisation engine (mixed-integer linear program).** Given those
   predictions, picks the squad, starting XI, captain and transfers that maximise
   expected points subject to FPL's hard rules — across a multi-week horizon, so
   it can bank transfers, spend them, or take a hit when (and only when) the maths
   justifies it.

> The two are independent: you can change the model without touching the
> optimiser, and vice versa. "An ML algorithm that solves the linear programming"
> really means *ML feeds the LP* — keeping them apart is what makes the system
> debuggable.

A **Claude advisor** then reads the optimiser's output alongside the latest
football news and returns a plain-English recommendation with a **0–100** "how
strongly to act" score.

## Modelling opponents by characteristic

Instead of encoding the 20 opponents as 20 separate labels (sparse, and useless
the moment a team is promoted or changes manager), each opponent is described by a
vector of defensive traits — xG conceded, shots and shots-on-target conceded,
defensive-action volume, aerial strength, set-piece concession, press intensity,
possession. A gradient-boosted model learns the *interactions* (a given player
profile over- or under-performing against a given opponent profile — the "Haaland
against a deep low-block" effect), which **transfers to teams it has never seen**.
Alongside it, the teams are clustered into a few defensive archetypes purely for
human-readable explanation and for Claude's narrative.

## Decisions locked

| Decision | Choice |
|---|---|
| Language | **Python** |
| Data | **FPL API** (players, prices, fixtures, results, your squad — incl. per-player xG and CBIT defensive stats), **filed per season** in a season store so nothing is overwritten. Opponent stats from FPL's own results by default; FBref (needs Chrome) and Understat (PPDA, often blocked) are optional backends. |
| Seasons | Players are linked across seasons by FPL's **permanent player code** (ids change every season), with name + date of birth + nationality as fallbacks. Last season enters as a recency-weighted **prior** and as down-weighted training rows. |
| Planning | **Multi-week from v1** (not a later add-on) |
| Claude advisor | **Anthropic API with web search** for live news |
| Model | Gradient-boosted trees (scikit-learn HistGradientBoosting): a minutes model and a points model, `xp = P(plays) × points if he plays` |
| Interface | One **self-contained HTML report** (Plotly, works offline), built each week by `scripts/run_week.py` |

## Verified facts the build rests on

- **2026/27 season:** 22 Aug 2026 → 30 May 2027, 38 gameweeks. Under way: GW7 had
  finished at the last pull (24 Oct 2026). The live API serves only the current
  season, so 2025/26 is kept frozen in the season store (full-season totals for all
  841 players; per-match history for the top 50 only). Promoted: Coventry City,
  Hull City, Ipswich Town; relegated: Burnley, West Ham, Wolves.
- **Squad:** 15 players (2 GK / 5 DEF / 5 MID / 3 FWD), £100m budget, max 3 per club.
- **Transfers:** 1 free per gameweek, bank up to 5, retained through chips; extra
  transfers cost −4 each.
- **Chips (2025/26 baseline):** two sets of Wildcard / Free Hit / Bench Boost /
  Triple Captain, first set expiring at the GW19 deadline; no Assistant Manager chip.
- **Scoring** includes the 2025/26 *defensive contribution* points (defenders: 10
  combined clearances/blocks/interceptions/tackles → 2 pts; mid/fwd: 12 incl. ball
  recoveries → 2 pts).

> **Confirmed:** `config.py` matches FPL's own published 2026/27 rules (its
> `game_config`); one stale rule was fixed: a goalkeeper goal is worth **10** points.
> `ingest/rules_check.py` re-checks after every pull and warns if FPL changes a rule.

## Build checklist

- [x] **0 — Scaffold & docs.** Repo structure, `config.py` (all rules + tunable
  dials), this charter, the docs framework, a passing config test.
- [x] **1 — Data ingestion.** FPL API client (cached fetch + pure parsing); season
  store (file each season, freeze finished ones); cross-season player and club
  links; opponent-stats adapter (FPL results / FBref / Understat); team-name
  reconciliation; xG and CBIT defensive fields confirmed on the FPL endpoint.
- [x] **2 — Feature engineering.** Exponential recency weighting; opponent
  characteristic vectors + archetypes; player×opponent interactions; leakage-free
  cold-start priors; last season as a prior; one multi-season table with sample
  weights; player types (for the report).
- [x] **3 — Prediction model.** Minutes model (P(plays), P(60+)) and a points model
  for when he plays: gradient-boosted trees (scikit-learn's HistGradientBoosting,
  the LightGBM algorithm without its macOS system dependency), sample-weighted
  across seasons. Walk-forward backtest: its top-11 picks beat recency-weighted form
  by ~11 points a gameweek. FPL injury flags applied to predictions.
- [x] **4 — Optimisation.** Single-gameweek MILP (PuLP + bundled CBC): squad, XI,
  captain and vice under all hard rules.
- [x] **5 — Multi-week planning.** One MILP over a rolling 5-week horizon from your
  real squad (public FPL data: picks, bank, estimated free transfers, FPL selling
  prices); bank vs spend vs −4 hit only when the gain clears the threshold.
  *Not yet:* chip timing.
- [x] **6 — Claude advisor.** Claude Opus 5.5 + web search → a strict structured
  answer: 0–100 action score, verdict, flagged players with sources; news adjusts,
  never overrides, the maths. Refusal fallback on; never crashes the run.
- [x] **7 — Interface.** `scripts/run_week.py` (one command a week) → one
  self-contained HTML report: this week's plan, the news check, the best squad from
  scratch, predictions, the model check, and the player analysis.

## What's next
- **Chips:** plan the wildcard, free hit, bench boost and triple captain.
- **Full 2025/26 match data** from a public archive (only the top 50 players' matches
  were saved before the API moved on), for a stronger, less biased model.
- **Richer opponent stats** (team xG conceded, pressing) if a source becomes
  reachable; the opponent adapter takes it without other changes.
