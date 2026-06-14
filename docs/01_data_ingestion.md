# 01 — Data ingestion

**Goal of this section:** land the raw data we need, reliably and politely, and
get the sources speaking the same language. Turning these tables into model
*features* is Section 2 — here we just fetch, tidy and cache.

## The two sources, and why we split the work this way

**FPL API** (`fantasy.premierleague.com/api`, public, no login) gives everything
player- and game-related: every player with price, position, ownership, form and
status; per-match history; all fixtures with FPL's difficulty rating (FDR); and
your own squad. It now also carries **expected goals / assists** per player *and*
the new **defensive-contribution stats** (clearances, blocks, interceptions,
tackles, recoveries) the 2025/26 rules reward — confirmed present for all ~840
players. So the *player* side of the model needs no scraping at all.

**Understat** gives the one thing the FPL API doesn't: **how each team plays and
how it concedes**. It is our primary advanced-stats source.

The deliberate choice: **opponents are modelled at team level, players at player
level.** The only thing to reconcile between sources is *team names* (20 of them),
not thousands of players.

## Why Understat, not FBref (a foundation decision)

We originally planned to use FBref via `soccerdata`. On a real run that failed
with *"Chrome not found"*: FBref now sits behind Cloudflare, so `soccerdata` drives
a **headless Chrome browser** to read it — heavy, fragile, and dependent on having
Google Chrome installed.

Understat instead embeds its data as JSON inside a `<script>` tag, so a single
plain **HTTP GET** is enough (no browser). It also happens to give *better*
defensive-style signals than FBref's raw action counts:

| Understat field | meaning |
|---|---|
| `xGA`, `npxGA` | expected goals conceded — how leaky a team is |
| `ppda` | passes allowed per defensive action — their pressing intensity (lower = presses harder) |
| `ppda_allowed` | the opponent's PPDA against them — how much they sit off / cede the ball |
| `deep_allowed` | opponent passes completed near their goal — territory conceded |

FBref remains available as an **optional** secondary source in `fbref.py` for
anyone who wants its detailed defensive-action tables and is willing to install
Chrome (`brew install --cask google-chrome`).

## FPL specifics worth knowing
- **Prices are in tenths.** `now_cost = 151` → £15.1m. Converted at ingestion.
- **Positions are integers.** `element_type` 1/2/3/4 → GK/DEF/MID/FWD (FPL writes
  "GKP"; we normalise to "GK").
- **Some numbers arrive as strings** (e.g. `expected_goals`); we coerce them.
- **Resilient parsing.** Every optional column fills with `NA` if absent, so a
  quiet change to FPL's payload degrades gracefully instead of crashing.
- **"Free transfers available this week" isn't in the API** — it's inferred from
  transfer history (Section 5). `my_squad()` returns picks, bank, value and the
  captain flags.

## Understat specifics worth knowing
- **One GET per league-season**, cached to `data/cache/understat/` for a day.
- **Season = start year.** `to_understat_season("2025-26")` → `"2025"`.
- **Embedded-JSON decode.** The page stores data as `JSON.parse('<hex-escaped
  UTF-8>')`; `parse_team_matches()` extracts and decodes it. Fixed, known columns
  come out: `team, date, home_away, xg, xga, npxg, npxga, ppda, ppda_allowed,
  deep, deep_allowed, scored, missed (= goals conceded), xpts, result`.

## Reconciling team names
`team_map.reconcile(fpl_names, understat_names)` normalises both lists
(lowercasing, fixing apostrophes, mapping variants like "Manchester United" →
"Man Utd") and returns the matches **plus** any team on only one side. Empty
`fpl_only`/`fbref_only` lists mean everything lined up; a non-empty list is your
cue to add an alias (usually after promotion/relegation).

## Running it (on your machine)
```python
from fpl_optimiser.ingest import fpl_api, understat, team_map

players  = fpl_api.players()           # tidy player table, price in £m
fixtures = fpl_api.fixtures()          # with team names + FDR

tm = understat.team_matches("2025-26") # per-team per-match xGA / PPDA / deep
ts = understat.team_season("2025-26")  # season aggregates per team

rec = team_map.reconcile(players["team_name"].unique(), tm["team"].unique())
print(rec["fpl_only"], rec["fbref_only"])   # should both be empty
```

## Tests
`tests/test_ingest.py` runs the parsing logic on mock payloads (no network):
FPL price/position/team derivation, numeric coercion, missing-column handling,
fixture joins; the **Understat embedded-JSON extract+decode** on a synthetic page;
season-code helpers; and team reconciliation across FPL/FBref/Understat spellings.

## What's deferred to Section 2
Recency weighting, building the per-opponent characteristic vector from these
tables, clustering teams into defensive archetypes, and player×opponent
interaction features.
