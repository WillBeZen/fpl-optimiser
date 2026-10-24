# 06 — Claude advisor: the news check

**Goal:** before you act on the plan, check what the maths can't see (injury news
from press conferences, rotation hints, suspensions, late fitness tests) and get a
**0–100 score for how strongly to act** on this week's moves.

**News adjusts, never overrides, the maths.** The advisor flags players and can
suggest an adjustment ("if X is ruled out, move the armband"); it never rewrites the
plan. The optimiser's numbers stay the decision; the advisor tells you how much to
trust them this week.

```bash
# in .env (never commit it):
ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_week.py --open      # the news check runs as step 5
```

## How it works (`fpl_optimiser/claude/advisor.py`)
`brief()` writes the prompt: the gameweek and deadline, this week's transfers out and
in, free transfers and hits, the expected-points gain over making no transfers, the
captain and vice, and every squad player with **FPL's own status flag and news text**.

`advise()` makes **one request** to `config.ADVISOR_MODEL` (Claude Opus 5.5) with:
- **adaptive thinking** at `config.ADVISOR_EFFORT` (`high`);
- **Anthropic's web search tool** (`web_search_20260209`, up to `ADVISOR_MAX_SEARCHES`
  searches). Claude decides what to look up;
- a **strict `submit_advice` tool** whose arguments *are* the answer. Strict mode
  guarantees they match the schema: `action_score` (0–100), `verdict`
  (`act` / `adjust` / `wait`), `summary`, captain check, `player_flags` (status
  fit / doubt / out / unknown, a note and the **source URL** each claim rests on),
  `adjustments` and `risks`. The call is read, never executed.

The system prompt tells Claude to base every injury or rotation claim on a source it
found, and to say "unknown" rather than guess.

**Robustness.** A long search turn can pause (`pause_turn`); it is resumed, up to 4
times. If Claude answers in prose instead of calling the tool, it gets one nudge.
**Server-side refusal fallback** is on (`fallbacks: "default"`, beta
`server-side-fallback-2026-07-01`): if a safety classifier declines the request, the
API re-runs it on Anthropic's recommended fallback model. Any API trouble (missing or
rejected key, rate limit, network) returns an `Advice` with `error` set, so the
weekly run still produces its report, showing "Advisor unavailable: <reason>".

**Cost.** One request a week with a handful of searches. Change `ADVISOR_MODEL` or
`ADVISOR_EFFORT` in `config.py` to trade depth for cost.

## In the report
The "News check (Claude)" section shows the score, the verdict and summary, a table
of flagged players with links to their sources, suggested adjustments, risks, and
the list of pages searched. Everything Claude returns, and every URL, is treated as
untrusted text: it's escaped, and only `http(s)` links become clickable.

## Tests
`tests/test_advisor.py` (a fake client, no network or cost): the brief lists moves,
captain and FPL flags; the advice is read from the `submit_advice` call (score clamped
to 0–100); the request is well-formed (model, effort, web search, strict tool, auto
tool choice, fallback beta); a paused search turn is resumed; a prose answer gets one
nudge and then gives up; refusals and network errors never crash the run.
