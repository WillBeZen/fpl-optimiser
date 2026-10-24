# CLAUDE.md — working context for this repo

You are the coding agent for **fpl-optimiser**. Read this before editing. Keep the
test suite green and don't break the invariants below.

## Goal

A weekly Fantasy Premier League decision system with **two separable engines**:

1. **Prediction** (ML/stats) — estimate each player's expected FPL points for the
   upcoming gameweek(s). Recency weighting and opponent "characteristics" live here.
2. **Optimisation** (a mixed-integer linear program) — pick the 15-man squad,
   starting XI, captain and transfers that maximise predicted points under FPL's
   rules, across a **multi-week** horizon.

A Claude advisor then interprets the optimiser's output against live football news
and returns a recommendation with a 0–100 "how strongly to act" score.

Full plan and rationale: `docs/00_charter.md`. One plain-English doc per section
lives in `docs/` — **update the matching doc whenever you change a section.**

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # NOT inside iCloud — see Gotchas
pip install -r requirements.txt
python -m pytest -q                            # MUST stay green (33 tests today)
python scripts/verify_setup.py                 # live health check (FPL + opponent source)
python scripts/build_features.py --limit 50    # quick training-table build
python scripts/build_features.py               # full season -> data/processed/training_table.csv
```

Run everything from the repo root (a root `conftest.py` puts it on `sys.path`, so
`import config` and `import fpl_optimiser` work without an install).

## Repo layout

```
config.py                     SINGLE SOURCE OF TRUTH: FPL rules + strategy knobs
conftest.py                   makes the repo root importable
fpl_optimiser/
  ingest/
    fpl_api.py                FPL API client (cached fetch + PURE parse_* fns)   [done]
    opponent_stats.py         source-agnostic opponent table + auto-fallback     [done]
    fbref.py                  FBref via soccerdata (needs Chrome) — a backend     [done]
    understat.py              Understat plain-HTTP (PPDA) — a backend, often blocked [done]
    team_map.py               FPL<->source team-name reconciliation              [done]
  features/
    recency.py                exponential decay weighting                        [done]
    opponent.py               opponent profiles + k-means archetypes             [done]
    build.py                  leakage-safe (player, gameweek) training table     [done]
  model/                      minutes model, points model, backtest              [Section 3 - stubs]
  optimise/                   squad MILP (S4) + multi-week planner (S5)           [stubs]
  claude/advisor.py           Claude advisor + news -> score/100 (S6)            [stub]
  app/dashboard.py            Streamlit UI (S7)                                   [stub]
scripts/
  verify_setup.py             environment + live-source health check
  build_features.py           builds & caches the training table
docs/                         00_charter + one doc per section
tests/                        pytest; run on MOCK data, never hit the network
data/                         raw / processed / cache (git-ignored)
```

## Architecture invariants (do not violate)

1. **No leakage.** Every feature for gameweek *t* must use only data from rounds
   **strictly before *t***. See `features/build.py` (`weighted_form_asof`,
   `team_defence_asof`). Any new feature must obey this, and needs a test proving it.
2. **`config.py` is the single source of truth.** FPL rules (squad size, budget,
   scoring, transfer costs) and tunable knobs (half-life, planning horizon, hit
   threshold) live there. Never hardcode these elsewhere — import from `config`.
3. **Opponent source is abstracted.** Feature/model code calls
   `opponent_stats.opponent_stats(season)` only. Never import `fbref`/`understat`
   directly outside `opponent_stats.py`.
4. **Fetch vs parse are separate in ingest.** HTTP/browser calls are thin and
   cached; parsing is pure and unit-tested on mock payloads. Preserve this split so
   tests stay offline.
5. **Tests never touch the network.** Use mock payloads / `monkeypatch`. Keep the
   suite green; add tests with new logic.
6. **The two engines stay separable.** Prediction outputs expected points;
   optimisation consumes them. Don't entangle them.

## Data sources (reality, learned the hard way)

- **FPL API** (`fantasy.premierleague.com/api`, no auth) is primary. It has all
  player data incl. per-player **xG/xA and CBIT defensive stats**, fixtures with
  FDR, and squads. No scraping needed for players.
- **Advanced team stats** come through the `opponent_stats` adapter:
  - `fbref` (default) — via `soccerdata`, which drives **headless Chrome**
    (`brew install --cask google-chrome`). Its `schedule()` has scores but **no xG**
    in this version, so we derive goals-conceded; `xga_per_game` is NaN for now.
  - `understat` — plain HTTP, adds **PPDA**, but is currently **Cloudflare-blocked**
    for plain requests. Adapter auto-falls-back.
- Be polite: caching is on everywhere; don't disable soccerdata's rate limiting.

## Conventions

- British spelling in identifiers/docs: **optimise**, **archetype**, etc.
- Flat package layout; type hints; module docstrings explain what+why for a learner.
- Resilient parsing: fill missing optional columns with `NA` rather than crashing.
- Prices from FPL are in tenths (`now_cost 151` = £15.1m) — convert at ingestion.
- Positions: FPL `element_type` 1/2/3/4 -> GK/DEF/MID/FWD (FPL's "GKP" -> "GK").
- Small, logical commits; run `pytest` before committing.

## Gotchas already discovered (don't re-learn these)

- **Double gameweeks**: a team can play twice in one round — key on unique
  `(team, round)`, not per fixture (see `team_defence_asof`).
- FPL "free transfers available this week" is **not** in the API — infer from
  transfer history (Section 5).
- `dash` (macOS `/bin/sh`) has no brace expansion; use explicit paths in scripts.
- **iCloud vs the venv**: this repo sits in iCloud-synced `~/Documents`, and iCloud
  offloaded 98% of `.venv` (files go `dataless`), so imports stalled for minutes and
  pytest looked hung. Keep the venv outside iCloud (e.g. `~/.venvs/fpl-optimiser`).
- **seleniumbase hijacks pytest**: soccerdata pulls in seleniumbase, which
  auto-registers pytest plugins. `pytest.ini` sets `--disable-plugin-autoload`.
- **Tests are network-guarded**: `tests/conftest.py` makes any DNS/socket call raise.
- **The FPL cache is season-blind**: the key has no season and a 6–12 h TTL. Since
  Aug 2026 the live API serves 2026/27, so a refresh overwrites the 2025/26 files.
  A frozen copy lives in `data/raw/fpl_snapshot_2025-26/`.
- The FPL API needs a `User-Agent` header and a trailing slash on endpoints.

## Roadmap

- [x] 0 Scaffold · [x] 1 Data ingestion · [x] 2 Feature engineering
- [ ] **3 Prediction model** — minutes model, then points model (LightGBM,
  recency-weighted samples), then a **walk-forward** backtest vs a naive baseline.
- [ ] 4 Optimisation — single-GW MILP (PuLP): squad + XI + captain under all rules.
- [ ] 5 Multi-week planning — rolling horizon; bank vs spend vs −4 hit only when the
  gain over the horizon clears `config.HIT_THRESHOLD_POINTS`; chip timing.
- [ ] 6 Claude advisor — Anthropic API + web search -> JSON recommendation + 0–100
  score; news adjusts, never overrides, the maths.
- [ ] 7 Interface — Streamlit dashboard.

## Working style

Sections are designed one at a time (the human brings tested code/diffs from a
planning chat). When you apply a change: put files at the paths above, run
`pytest`, and skim edits to core modules before committing. The tests are the
safety net — if a change would require weakening a test, stop and flag it.v