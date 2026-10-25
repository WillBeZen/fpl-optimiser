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
and returns a recommendation with a 0–100 "how strongly to act" score. Everything lands
in a tabbed, self-contained HTML dashboard that shows each stage of both engines.

Full plan and rationale: `docs/00_charter.md`. One plain-English doc per section
lives in `docs/` — **update the matching doc whenever you change a section.**

## How to run

```bash
source .venv/bin/activate                      # -> ~/.venvs/fpl-optimiser (see Gotchas)
pip install -r requirements.txt
python -m pytest -q                            # MUST stay green (142 tests today)
python scripts/run_week.py --open              # THE weekly run: pull -> predict -> plan -> advise -> report
python scripts/run_week.py --no-pull --no-advisor --quick   # fast re-run: no pull, no API spend, no what-ifs
python scripts/pull_fpl.py                     # just file the live season into data/raw/fpl/<season>/
python scripts/build_features.py               # just the multi-season training table -> data/processed/
python scripts/make_report.py --open           # just the analysis report
python scripts/verify_setup.py                 # live health check (FPL, opponent source, rules)
python scripts/check_claude.py                 # one tiny Claude request: is the key working?
```

Run everything from the repo root (a root `conftest.py` puts it on `sys.path`, so
`import config` and `import fpl_optimiser` work without an install).

## Repo layout

```
config.py                     SINGLE SOURCE OF TRUTH: FPL rules + strategy knobs
conftest.py                   makes the repo root importable
fpl_optimiser/
  ingest/
    fpl_api.py                FPL API client (fetch + PURE parse_* fns)          [done]
    season_store.py           files each season in data/raw/fpl/<season>/; freeze [done]
                              (played_only: only FINISHED matches count as played)
    player_link.py            same player across seasons: code -> name+dob -> dob+region+surname [done]
    opponent_stats.py         source-agnostic opponent table + auto-fallback     [done]
    fbref.py                  FBref via soccerdata (needs Chrome) — a backend     [done]
    understat.py              Understat plain-HTTP (PPDA) — a backend, often blocked [done]
    team_map.py               FPL<->source team-name reconciliation              [done]
    my_team.py                your squad, bank, free transfers, selling prices, chips left [done]
    rules_check.py            config.py vs FPL's published rules                 [done]
  features/
    recency.py                exponential decay weighting                        [done]
    opponent.py               opponent profiles + k-means archetypes             [done]
    build.py                  leakage-safe (player, gameweek) training table     [done]
    prior_season.py           last season per player, as a prior                 [done]
    multi_season.py           every filed season stacked, with sample weights    [done]
    player_profiles.py        per-90 player profiles + player types (report)     [done]
  model/
    inputs.py                 the feature list both models use                   [done]
    minutes.py                P(plays), P(60+)                                   [done]
    points.py                 xp = p_play x points-if-he-plays                   [done]
    events.py                 event model: team ratings -> fixture goals -> player events [done]
    backtest.py               walk-forward backtest: blend, points, events, baselines [done]
    predict.py                forecast(): blended xp for the next N gameweeks    [done]
    explain.py                numbers behind the report's "The model" tab        [done]
    evaluation.py             THE honest replay: every past deadline as it stood, all methods [done]
    accuracy.py               accuracy views + data checks on the replay        [done]
    forecast_log.py           files each run's next-gameweek forecast            [done]
  optimise/
    squad.py                  single-gameweek MILP: 15, XI, captain, vice        [done]
    planner.py                multi-week MILP: transfers, banking, hits, chips   [done]
    pool.py                   drops dominated players before solving             [done]
    whatif.py                 transfer ladder, budget value, alternatives, chip grid, ... [done]
  claude/advisor.py           Claude news check -> 0-100 score (web search)     [done]
  app/
    charts.py                 Plotly figure builders (light + dark, validated palette) [done]
    html.py                   Section/Report/render, page CSS + JS, data_table   [done]
    common.py                 WeekResults, Context, shared lookups               [done]
    tab_*.py                  one module per dashboard tab (week, squad, fixtures,
                              scout, model, accuracy, planner)                   [done]
    report.py                 build_report: assembles the tabs (S7)              [done]
    glossary.py               the report's glossary (values read from config)    [done]
scripts/
  run_week.py                 the weekly run (everything, one command)
  pull_fpl.py                 files the live season (polite, resumable)
  build_features.py           builds the multi-season training table
  make_report.py              writes reports/fpl_report.html
  verify_setup.py             environment + live-source health check
  check_claude.py             one tiny Claude request to test the API key
docs/                         00_charter + one doc per section
tests/                        pytest; MOCK payloads / tiny synthetic seasons, never the network
data/                         raw (filed seasons) / processed / cache (git-ignored)
reports/                      generated HTML (git-ignored)
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
7. **Ids are per season; `code` is forever.** FPL re-assigns player and team `id`s
   every season. Never join two seasons on `id`: link players with
   `player_link.link_players` (permanent `code` first) and clubs on their `code`.
8. **Filed seasons are immutable.** Raw data lives in `data/raw/fpl/<season>/`; a
   frozen season is never rewritten. Read seasons through `season_store.load`.

## Data sources (reality, learned the hard way)

- **FPL API** (`fantasy.premierleague.com/api`, no auth) is primary. It has all
  player data incl. per-player **xG/xA and CBIT defensive stats**, fixtures with
  FDR, and squads. No scraping needed for players.
- The live API serves **only the current season**. 2025/26 is frozen in the store
  (season totals for all 841 players; per-match history for the top 50 only).
- **Opponent (team) stats** come through the `opponent_stats` adapter:
  - `fpl` (default) — goals conceded from the season's filed FPL results; no browser.
  - `fbref` — via `soccerdata`, which drives **headless Chrome**. Its `schedule()`
    has scores but **no xG** in this version, i.e. nothing `fpl` doesn't give.
  - `understat` — plain HTTP, adds **PPDA**, but often **Cloudflare-blocked**.
- Be polite: the pull uses one keep-alive session, pauses between players and backs
  off on 429/5xx; don't disable soccerdata's rate limiting if you use FBref.

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
- **iCloud vs Python**: the repo used to live in iCloud-synced `~/Documents`;
  iCloud offloaded 98% of `.venv` (files go `dataless`), so imports stalled for
  minutes and pytest looked hung. Even Python's start-up scan of an iCloud working
  directory can stall. The repo now lives in `~/Developer`; `.venv` is a symlink to
  `~/.venvs/fpl-optimiser`. Keep both out of iCloud.
- **seleniumbase hijacks pytest**: soccerdata pulls in seleniumbase, which
  auto-registers ~7 pytest plugins. `pytest.ini` sets `--disable-plugin-autoload`.
- **Tests are network-guarded**: `tests/conftest.py` makes any DNS/socket call raise.
- **The old FPL cache (`data/cache/fpl/`) is season-blind** (no season in the key,
  6–12 h TTL). Only `verify_setup.py` and the `fpl_api` convenience functions use
  it; everything that builds features reads the season store instead.
- **Accents in names**: Unicode can't decompose ø/æ/ß/ł, so stripping accents
  deletes them (Ødegaard -> "degaard"); `player_link.normalise_name` spells them
  out first.
- **Plotly chart ids**: a chart's DOM id must differ from its section's id
  (`getElementById` returns the first match and the chart lands in the wrong place);
  the report prefixes chart ids with `fig-`.
- The FPL API needs a `User-Agent` header and a trailing slash on endpoints.
- **LightGBM needs `libomp` on macOS**; the models use scikit-learn's
  HistGradientBoosting instead (same algorithm, no system dependency).
- **scikit-learn's trees refuse an all-missing column** ("window shape cannot be
  larger than input array shape"); `model/inputs.fit_matrix` makes such columns
  constant. Below 160 rows the models fall back to plain averages.
- **PuLP 4 drops the bundled CBC solver**; requirements pin `pulp<4`, and
  `squad.solve()` silences the deprecation notice.
- **Claude Sonnet 5.5 rejects forced `tool_choice`** (`any`/`tool`): the advisor uses
  `auto` + a strict `submit_advice` tool + one nudge. Web search can end a turn with
  `pause_turn`; resend `[user, assistant(content)]` to resume.
- **An API key can be valid but unusable**: with no credit on the account every
  request is a 400 ("Your credit balance is too low ..."). `check_claude.py` tells you
  in one cheap call; `advisor.api_message` turns API errors into that plain sentence.
- **Mid-gameweek, FPL lists unplayed matches** (0 minutes) in every player's history.
  `season_store.played_only` keeps only finished fixtures; never bypass it, or form and
  the minutes model are corrupted for every club still to play.
- **Plotly `add_annotation` in a loop is quadratic** (it re-validates every earlier
  annotation): a 15 x 20 heatmap took 90 s. Build a list and set `layout.annotations`.
- **DOM ids must be unique in the report** (sections, tables, charts): a clash breaks
  the page script for every tab. `tests/test_report.py` checks it.
- **The planner prunes dominated players** (`optimise/pool.py`); pass `prune=False` to
  check an answer against the full problem (a test does).
- **The real Anthropic key costs money**: verify runs with `--no-advisor`; offline tests
  use a fake client. The last successful check is reused from `reports/last_advice.json`.
- **FPL's season totals (bootstrap `total_points`) count matches still in play**; the
  filed match records lag until the next refresh. Check finished gameweeks against
  `event/<gw>/live` (filed as `event-live_<gw>.json`), never against bootstrap totals.
- **Every model input must be knowable at the deadline.** The opponent "style" input was
  clustered on season-to-date goals conceded (for a past match that includes the match
  itself) and inflated the backtest's top-11 by ~14 points; it is no longer an input
  (`model/inputs.py`). Never add a feature built from season-level or current snapshots.
- **Two code paths, one answer:** numeric inputs are rounded (`inputs.ROUND_TO`) so the
  training table and the forecast can't put a value on different sides of a tree split.
- **Evaluate honestly** with `model/evaluation.py` (pools, prices, features, blend
  weight all as of each deadline). Don't tune a setting on the weeks you then quote as
  evidence; `EVENT_MODEL_WEIGHT` (0.5) was, and is flagged as in-sample.
- **The accuracy replay can't apply past injury flags** (FPL doesn't archive them);
  `forecast_log` files each week's real forecast so later weeks can compare it.
- **Free transfers aren't in FPL's API**: `my_team.free_transfers` replays your
  history; one-off FPL top-ups aren't visible, so `run_week.py --free-transfers N`
  overrides it.
- **Chips make CBC slow** (267 s for the first version, no answer): it struggles to *find* a
  plan once chip decisions are in. `planner._solve_with_chips` hands it one: the no-chip
  plan, then the chips worth playing on its squads, then the best plan with them. The full
  problem then stops after `planner.CHIP_SOLVE_SECONDS`; `Plan.stats["proven_optimal"]`
  says whether it finished. Forced chips (`play_chips`, the what-ifs) skip all that.
- **CBC 2.10 misreads a warm start in a maximisation** (it reads the starting objective
  with the wrong sign, so the start is useless). `planner._solve_from` solves the
  minimisation of minus the objective instead.
- **A free hit doesn't depend on the plan**: with prices held fixed, transfers keep bank +
  squad value constant, so each week's free-hit squad is a one-week `pick_squad` made
  up front. Don't model it as a second squad inside the MILP: that's 2,880 more yes/no
  variables and 3x slower.

## Roadmap

- [x] 0 Scaffold · [x] 1 Data ingestion · [x] 2 Feature engineering
- [x] 3 Prediction model · [x] 4 Optimisation · [x] 5 Multi-week planning
- [x] 6 Claude advisor · [x] 7 Interface (`scripts/run_week.py` + tabbed HTML dashboard)
- [x] Event model + blend (backtested), return dates from injury news, planner what-ifs
- [x] **Chips** — wildcard / free hit / bench boost / triple captain planned in planner.py
      (thresholds `CHIP_THRESHOLD_POINTS`, the GW19 split, a chip × gameweek what-if).
- [ ] **Full 2025/26 match data** from a public archive (only the top 50 were filed).
- [ ] Richer opponent stats (team xGA, PPDA) if a source becomes reachable.

## Working style

Sections are designed one at a time (the human brings tested code/diffs from a
planning chat). When you apply a change: put files at the paths above, run
`pytest`, and skim edits to core modules before committing. The tests are the
safety net — if a change would require weakening a test, stop and flag it.v