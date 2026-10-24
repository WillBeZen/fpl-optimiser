# 01 — Data ingestion

**Goal of this section:** land the raw data we need, reliably and politely, file
it by season so nothing is ever overwritten, and get the sources speaking the same
language. Turning these tables into model *features* is Section 2 — here we just
fetch, tidy, file and link.

## The sources, and why we split the work this way

**FPL API** (`fantasy.premierleague.com/api`, public, no login) gives everything
player- and game-related: every player with price, position, ownership, form and
status; per-match history; all fixtures and results with FPL's difficulty rating
(FDR); and your own squad. It also carries **expected goals / assists** per player
*and* the 2025/26 **defensive-contribution stats** (clearances, blocks,
interceptions, tackles, recoveries). So the *player* side needs no scraping at all.

**Opponent (team) stats** come through one adapter, `opponent_stats(season)`, with
swappable backends (config `ADVANCED_STATS_SOURCE`):

| backend | what it gives | catch |
|---|---|---|
| `fpl` (default) | goals conceded per game, from the season's own filed FPL results | no team xG / pressing data |
| `fbref` | the same goals conceded (this soccerdata version's schedule has **no xG**) | needs `soccerdata` + Google Chrome (FBref is behind Cloudflare, so soccerdata drives a headless browser) |
| `understat` | xGA, PPDA (pressing), deep completions | often Cloudflare-blocked for plain HTTP |

FBref was the planned source, but right now it gives exactly what FPL's results
already give, at the cost of a browser. So `fpl` is the default: no browser, and
always the right season. If one backend errors, the adapter tries the next. When a
richer source becomes reachable, add its columns and nothing downstream changes.

The deliberate choice: **opponents are modelled at team level, players at player
level.** Only *team names* (20 of them) ever need reconciling between sources.

## The season store — "file it, never overwrite it"

The live API only ever serves the **current** season, and FPL **re-assigns every
player and team `id` each season** (Haaland was id 430 in one season and 411 the
next). A plain response cache is therefore dangerous: when the season rolls over, a
refresh silently replaces last season's files with this season's, under the same
names and with different ids. So raw data is filed by season:

```
data/raw/fpl/<season>/bootstrap-static.json     players, teams, gameweeks (season totals)
                     /fixtures.json             every fixture and result
                     /element-summary_<id>.json each player's per-match history (+ history_past)
                     /FROZEN                    present => read-only
```

- `python scripts/pull_fpl.py` files the live season. The season is read from the
  data itself (GW1's deadline), never assumed. Re-runs are cheap: player files
  younger than `--ttl` hours are kept, so an interrupted pull just resumes.
- A finished season is **frozen**; `pull_live()` refuses to write into it. After
  every pull, any older season still unfrozen is frozen automatically, because the
  API will never serve it again.
- `season_store.load(season)` reads a filed season back as tidy tables (players,
  teams, fixtures, gameweeks, per-match history, `history_past`), and refuses a
  folder whose data belongs to a different season.

**What is filed today:** 2025/26 (frozen; full-season players, fixtures and results,
but per-match history for only the top 50 players, because that is all that was
fetched before the API moved on) and 2026/27 (all 668 players, refreshed by
`pull_fpl.py`).

## Linking seasons — the same player, two ids

`player_link.link_players(prev, curr)` matches last season's players to this
season's in three passes, strictest first. Each pass only sees players the earlier
passes left unmatched, and only one-to-one matches are accepted:

1. **`code`**: FPL's permanent player code (Opta's id, e.g. Raya = 154561). Exact. A
   code match whose birth dates disagree is rejected, not trusted.
2. **name + date of birth** (accents, case and punctuation ignored, so Ødegaard =
   Odegaard): catches a player whose code changed.
3. **date of birth + nationality (`region`) + surname**: catches a player whose
   listed name changed.

Anything ambiguous (two equally good candidates) is set aside in `rejected`, never
guessed. Whoever is left is a genuine arrival or departure. Clubs are linked the
same way on their permanent club `code` (`link_teams`): stayed / promoted / relegated.

`check_against_history_past()` is an independent check: each player's
element-summary carries `history_past`, his totals for earlier seasons tagged with
his permanent code, so the links can be checked against FPL's own record.

**2025/26 → 2026/27 on real data:** 481 players linked, all by `code`; FPL's own
history agrees on the code *and* the 2025/26 points for all 481; 0 rejected; 187
genuine arrivals (99 from promoted Hull, Coventry and Ipswich); 360 departures;
65 linked players changed club and 11 changed FPL position. Relegated: Burnley,
West Ham, Wolves.

## FPL specifics worth knowing
- **Ids are per season; `code` is forever.** Never join two seasons on `id`.
- **Prices are in tenths.** `now_cost = 151` → £15.1m. Converted at ingestion.
- **Positions are integers.** `element_type` 1/2/3/4 → GK/DEF/MID/FWD (FPL writes
  "GKP"; we normalise to "GK").
- **Some numbers arrive as strings** (e.g. `expected_goals`); we coerce them.
- **Resilient parsing.** Every optional column fills with `NA` if absent, so a
  quiet change to FPL's payload degrades gracefully instead of crashing.
- **Rate limits and update windows.** `fpl_api.fetch()` retries politely on 429/5xx,
  and the pull spaces out its ~700 requests over one keep-alive connection.
- **"Free transfers available this week" isn't in the API** — it's inferred from
  transfer history (Section 5). `my_squad()` returns picks, bank, value and the
  captain flags.

## Understat specifics (optional backend)
- **One GET per league-season**, cached to `data/cache/understat/` for a day.
- **Season = start year.** `to_understat_season("2025-26")` → `"2025"`.
- **Embedded-JSON decode.** The page stores data as `JSON.parse('<hex-escaped
  UTF-8>')`; `parse_team_matches()` extracts and decodes it.

## Reconciling team names
`team_map.reconcile(fpl_names, source_names)` normalises both lists (lowercasing,
fixing apostrophes, mapping variants like "Manchester United" → "Man Utd") and
returns the matches **plus** any team on only one side. A non-empty `fpl_only` /
`fbref_only` list is your cue to add an alias (usually after promotion/relegation).
With the default `fpl` backend the names already match.

## Running it (on your machine)
```bash
python scripts/pull_fpl.py               # file the live season (a few minutes the first time)
```
```python
from fpl_optimiser.ingest import season_store, player_link

prev, curr = season_store.load("2025-26"), season_store.load("2026-27")
res = player_link.link_players(prev.players, curr.players)
print(res.summary())                     # linked / by method / arrivals / departures
```

## Tests
- `tests/test_ingest.py` — FPL parsing on mock payloads (price/position/team,
  numeric coercion, missing columns, permanent codes), goals conceded from FPL
  results, the Understat embedded-JSON decode, season-code helpers, team-name
  reconciliation, and the FBref path with and without xG.
- `tests/test_season_store.py` — the season is read from the data, pulls resume,
  a frozen season is never overwritten, and a mislabelled folder is refused.
- `tests/test_player_link.py` — each linking pass, accent handling (Ødegaard),
  ambiguous and contradictory candidates are never linked, club links, and the
  `history_past` check.
