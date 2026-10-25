"""Claude advisor (Section 6) — reads the week's plan against the latest news.

The optimiser's maths decides the plan. This reads what the maths can't see
(press-conference injury news, rotation hints, suspensions, late fitness tests)
and returns a 0-100 score for how strongly to act on this week's moves, with
reasons and sources. **News adjusts, never overrides, the maths:** the advisor
flags players and suggests adjustments; it never rewrites the plan itself.

One request to Claude (config.ADVISOR_MODEL, adaptive thinking at
config.ADVISOR_EFFORT) with two tools:
  web_search      Anthropic's server-side search; Claude decides what to look up
  submit_advice   a strict client tool whose arguments ARE the structured answer
                  (strict mode guarantees they match the schema), called once at
                  the end. Its call is read, never executed.
Server-side refusal fallback is enabled (`fallbacks: "default"`): if a safety
classifier declines the request, the API re-runs it on Anthropic's recommended
fallback model instead of failing.

`brief()` turns the plan into the prompt; `advise()` makes the call. Both work
without a network in tests (pass a fake `client`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import config

SUBMIT = "submit_advice"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_MAX_CONTINUATIONS = 4      # web search can pause a long turn; resume at most this often

# Anthropic list prices for the cost readout: $ per million tokens (input, output),
# and $ per web search ($10 per 1,000). Thinking is billed as output; the pages a
# search returns arrive as input tokens.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}
SEARCH_PRICE = 10.00 / 1000


def api_message(exc) -> str:
    """The human part of an Anthropic API error ("Your credit balance is too low ..."),
    not the raw JSON body."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(getattr(exc, "message", exc))


def approx_cost(model: str, input_tokens: int, output_tokens: int, searches: int = 0) -> float | None:
    """Rough $ cost at list prices (None for a model without a listed price)."""
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6 + searches * SEARCH_PRICE


ADVICE_TOOL = {
    "name": SUBMIT,
    "description": "Submit your final assessment of this week's FPL plan. Call exactly once, "
                   "after you have checked the news.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "action_score": {"type": "integer", "description": "0 = don't act on these moves "
                             "this week (wait for news); 100 = act now with confidence."},
            "verdict": {"type": "string", "enum": ["act", "adjust", "wait"]},
            "summary": {"type": "string", "description": "2-4 plain sentences."},
            "captain_ok": {"type": "boolean"},
            "captain_note": {"type": "string"},
            "player_flags": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "player": {"type": "string"},
                    "status": {"type": "string", "enum": ["fit", "doubt", "out", "unknown"]},
                    "note": {"type": "string"},
                    "source": {"type": "string", "description": "URL the claim rests on, or ''"},
                },
                "required": ["player", "status", "note", "source"],
                "additionalProperties": False}},
            "adjustments": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action_score", "verdict", "summary", "captain_ok", "captain_note",
                     "player_flags", "adjustments", "risks"],
        "additionalProperties": False,
    },
}

SYSTEM = """You are the news desk for a Fantasy Premier League decision system.

An optimisation model has already chosen this week's transfers, starting XI and
captain from expected-points predictions. Your job is what the model cannot see:
current team news. For the players involved, search for the latest injury and
fitness news, suspensions, press-conference quotes about selection or rotation,
and any fixture changes, then judge how strongly the manager should act on the
plan before this gameweek's deadline.

- Score 0-100: 0 means don't make these moves yet (wait for news), 100 means act
  now with confidence. A sound plan with no worrying news should score high.
- verdict: "act" (go ahead), "adjust" (go ahead with a change you name in
  adjustments), or "wait" (hold the transfers until news is clearer).
- Don't rebuild the team. Suggest at most a few concrete adjustments.
- Every injury, suspension or rotation claim must rest on a source you found;
  put its URL in the flag. If you can't find news about a player, mark him
  "unknown" rather than guessing.
- Finish by calling submit_advice exactly once."""


@dataclass
class Advice:
    action_score: int
    verdict: str
    summary: str
    captain_ok: bool = True
    captain_note: str = ""
    player_flags: list[dict] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    model: str = ""
    searches: int = 0
    sources: list[dict] = field(default_factory=list)    # {"title", "url"} from the searches
    error: str = ""                                     # set when no advice could be produced
    input_tokens: int = 0                               # across every request this run made
    output_tokens: int = 0
    checked_at: str = ""                                # when the check ran, "YYYY-MM-DD HH:MM" UTC
    gameweek: int | None = None                         # the gameweek it was about

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def cost(self) -> float | None:
        """Approximate $ for this run: tokens plus web searches, at list prices."""
        return approx_cost(self.model, self.input_tokens, self.output_tokens, self.searches)


def unavailable(reason: str) -> Advice:
    return Advice(action_score=0, verdict="wait", summary="", error=reason)


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #
def _who(rows) -> str:
    return "; ".join(f"{r.web_name} ({r.team_name}, {r.position}, £{r.price:.1f}m)" for r in rows.itertuples())


def brief(week, players, deadline: str | None = None, baseline_xp: float | None = None,
          plan_xp: float | None = None, hits_cost: int = 0, note: str = "",
          from_scratch: bool = False) -> str:
    """The user message: this gameweek's moves, XI and captain, and FPL's own
    status/news for everyone involved. `week` is a planner.Week (or anything with
    the same fields: gw, sold, bought, team, free_transfers, hits).
    `from_scratch`: the squad was picked from scratch (a wildcard), not transferred."""
    info = players.set_index("id")
    lines = [f"Gameweek {week.gw}" + (f" (deadline {deadline} UTC)" if deadline else "") + "."]
    if from_scratch:
        lines.append("This squad was picked from scratch for this gameweek (as with a wildcard), "
                     "not reached by transfers: judge the picks and the captain.")
    elif len(week.bought):
        lines.append(f"Transfers OUT: {_who(week.sold)}")
        lines.append(f"Transfers IN: {_who(week.bought)}")
        lines.append(f"Free transfers available: {week.free_transfers}; hits taken: {week.hits}.")
    else:
        lines.append(f"No transfers this week: bank the free transfer ({week.free_transfers} available).")
    if plan_xp is not None and baseline_xp is not None:
        lines.append(f"Expected points over the planning horizon: {plan_xp:.1f} with this plan "
                     f"(minus {hits_cost} for hits) vs {baseline_xp:.1f} making no transfers.")
    team = week.team
    cap = team.loc[team["captain"], "web_name"].iloc[0]
    vice = team.loc[team["vice"], "web_name"].iloc[0]
    lines.append(f"Captain: {cap}. Vice-captain: {vice}.")
    lines.append("")
    lines.append("Squad, with FPL's own status flags (a available, d doubtful, i injured, "
                 "s suspended, u unavailable) and news:")
    for r in team.itertuples():
        p = info.loc[r.id] if r.id in info.index else None
        status = (p["status"] if p is not None else "") or "a"
        news = (p["news"] if p is not None and isinstance(p["news"], str) else "") or "-"
        lines.append(f"- {r.web_name} ({r.team_name}, {r.position}) {r.role}, expected "
                     f"{r.xp:.1f} pts; status {status}; news: {news}")
    if note:
        lines.append("")
        lines.append(f"Note: {note}")
    lines.append("")
    lines.append(f"Today is {datetime.now(timezone.utc):%d %B %Y}. Check the news, then call submit_advice.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #
def _sources(content) -> list[dict]:
    """Titles and URLs from web search results (an error result is an object, not a list)."""
    out = []
    for block in content:
        if getattr(block, "type", "") == "web_search_tool_result" and isinstance(block.content, list):
            out.extend({"title": getattr(r, "title", ""), "url": getattr(r, "url", "")} for r in block.content)
    return out


def advise(user_brief: str, client=None, model: str = config.ADVISOR_MODEL,
           effort: str = config.ADVISOR_EFFORT,
           max_searches: int = config.ADVISOR_MAX_SEARCHES) -> Advice:
    """Ask Claude to read the news and score the plan. Never raises for API trouble:
    returns an Advice with `error` set, so the rest of the weekly run still works."""
    import anthropic

    try:
        client = client or anthropic.Anthropic()
    except anthropic.AnthropicError as exc:
        return unavailable(f"no Anthropic credentials ({exc}); add ANTHROPIC_API_KEY to .env")

    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}, ADVICE_TOOL]
    messages = [{"role": "user", "content": user_brief}]
    sources: list[dict] = []
    searches = tokens_in = tokens_out = 0
    nudged = False
    for _ in range(_MAX_CONTINUATIONS + 2):
        try:
            response = client.beta.messages.create(
                model=model, max_tokens=16000, system=SYSTEM, tools=tools, messages=messages,
                tool_choice={"type": "auto"}, output_config={"effort": effort},
                betas=[FALLBACK_BETA], fallbacks="default")
        except anthropic.AuthenticationError:
            return unavailable("the Anthropic API key was rejected; check ANTHROPIC_API_KEY in .env")
        except anthropic.RateLimitError:
            return unavailable("rate limited by the Anthropic API; try again in a minute")
        except anthropic.APIStatusError as exc:
            return unavailable(f"Anthropic API error {exc.status_code}: {api_message(exc)}")
        except anthropic.APIConnectionError:
            return unavailable("couldn't reach the Anthropic API (network)")

        sources += _sources(response.content)
        usage = getattr(response, "usage", None)
        searches += getattr(getattr(usage, "server_tool_use", None), "web_search_requests", 0) or 0
        tokens_in += getattr(usage, "input_tokens", 0) or 0
        tokens_out += getattr(usage, "output_tokens", 0) or 0

        if response.stop_reason == "refusal":
            return unavailable("Claude declined this request")
        call = next((b for b in response.content
                     if getattr(b, "type", "") == "tool_use" and b.name == SUBMIT), None)
        if call is not None:
            a = call.input
            return Advice(action_score=max(0, min(100, int(a["action_score"]))), verdict=a["verdict"],
                          summary=a["summary"], captain_ok=a["captain_ok"], captain_note=a["captain_note"],
                          player_flags=a["player_flags"], adjustments=a["adjustments"], risks=a["risks"],
                          model=getattr(response, "model", model), searches=searches, sources=sources,
                          input_tokens=tokens_in, output_tokens=tokens_out,
                          checked_at=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}")
        if response.stop_reason == "pause_turn":           # a long search turn paused: resume it
            messages = [messages[0], {"role": "assistant", "content": response.content}]
            continue
        if response.stop_reason == "end_turn" and not nudged:  # answered in prose: ask once for the call
            messages = messages + [{"role": "assistant", "content": response.content},
                                   {"role": "user", "content": "Now call submit_advice with your assessment."}]
            nudged = True
            continue
        return unavailable(f"no advice returned (stop reason: {response.stop_reason})")
    return unavailable("the news search didn't finish")


# --------------------------------------------------------------------------- #
# Keeping the last check (so a re-run needn't pay for another)
# --------------------------------------------------------------------------- #
def save(advice: Advice, path) -> None:
    """Write a successful check to `path` (JSON) for reuse later in the same gameweek."""
    import json
    from dataclasses import asdict
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(advice), indent=1), encoding="utf-8")


def load(path, gameweek: int) -> Advice | None:
    """The saved check for `gameweek`, or None (missing, unreadable or another week)."""
    import json
    from dataclasses import fields
    from pathlib import Path

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("gameweek") != gameweek or data.get("error"):
        return None
    known = {f.name for f in fields(Advice)}
    return Advice(**{k: v for k, v in data.items() if k in known})
