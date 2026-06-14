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
| Data | **FPL API** (players, prices, fixtures, your squad — incl. per-player xG and CBIT defensive stats) + **Understat** (team xGA / PPDA / deep completions, fetched browserless). FBref via `soccerdata` is optional and needs Google Chrome. |
| Planning | **Multi-week from v1** (not a later add-on) |
| Claude advisor | **Anthropic API with web search** for live news |

## Verified facts the build rests on

- **2026/27 season:** 22 Aug 2026 → 30 May 2027, 38 gameweeks; fixtures released
  19 Jun 2026. It is currently the **off-season** — we train and backtest on the
  completed 2025/26 season and go live for GW1.
- **Squad:** 15 players (2 GK / 5 DEF / 5 MID / 3 FWD), £100m budget, max 3 per club.
- **Transfers:** 1 free per gameweek, bank up to 5, retained through chips; extra
  transfers cost −4 each.
- **Chips (2025/26 baseline):** two sets of Wildcard / Free Hit / Bench Boost /
  Triple Captain, first set expiring at the GW19 deadline; no Assistant Manager chip.
- **Scoring** includes the 2025/26 *defensive contribution* points (defenders: 10
  combined clearances/blocks/interceptions/tackles → 2 pts; mid/fwd: 12 incl. ball
  recoveries → 2 pts).

> **Not yet confirmed (revisit before GW1):** the exact 2026/27 rule set is not
> published yet (FPL usually announces in late July). Every rule lives in
> `config.py`, so updating is a one-file edit.

## Build checklist

- [x] **0 — Scaffold & docs.** Repo structure, `config.py` (all rules + tunable
  dials), this charter, the docs framework, a passing config test.
- [x] **1 — Data ingestion.** FPL API client (cached fetch + pure parsing) +
  FBref/Understat via `soccerdata`; team-name reconciliation; xG and CBIT
  defensive fields confirmed present on the FPL endpoint.
- [ ] **2 — Feature engineering.** Exponential recency weighting; opponent
  characteristic vectors + archetypes; player×opponent interactions; cold-start priors.
- [ ] **3 — Prediction model.** Minutes model, then points-component model
  (LightGBM, recency-weighted); walk-forward backtest vs a naive baseline.
- [ ] **4 — Optimisation.** Single-gameweek MILP (PuLP): squad, XI, captain under
  all hard rules.
- [ ] **5 — Multi-week planning.** Extend to a rolling horizon; bank vs spend vs
  −4 hit (only when the gain over the horizon clears the threshold); chip timing.
- [ ] **6 — Claude advisor.** Anthropic API + web search → structured JSON
  recommendation with a 0–100 action score; news adjusts, never overrides, the maths.
- [ ] **7 — Interface.** Streamlit dashboard over the modules.
