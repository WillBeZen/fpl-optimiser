"""Claude advisor (Section 6) — a fake client stands in for the API (no network)."""
from types import SimpleNamespace as NS

import anthropic
import httpx
import pandas as pd
import pytest

import config
from fpl_optimiser.claude import advisor

ADVICE = {"action_score": 150, "verdict": "act", "summary": "Go ahead.", "captain_ok": True,
          "captain_note": "Fit.", "adjustments": [], "risks": ["late fitness test"],
          "player_flags": [{"player": "Star", "status": "fit", "note": "trained",
                            "source": "https://example.com/news"}]}


def _response(stop_reason, *blocks, searches=0, tokens=(0, 0)):
    return NS(stop_reason=stop_reason, content=list(blocks), model=config.ADVISOR_MODEL,
              usage=NS(server_tool_use=NS(web_search_requests=searches),
                       input_tokens=tokens[0], output_tokens=tokens[1]))


def _search_result(*urls):
    return NS(type="web_search_tool_result",
              content=[NS(type="web_search_result", title=f"t{i}", url=u) for i, u in enumerate(urls)])


SUBMIT = NS(type="tool_use", name=advisor.SUBMIT, id="tu_1", input=ADVICE)
PROSE = NS(type="text", text="Here is my view...")


class FakeClient:
    """Returns the queued responses in order and records every request."""

    def __init__(self, *responses, error=None):
        self.responses, self.error, self.calls = list(responses), error, []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.responses.pop(0)


def _week():
    team = pd.DataFrame({"id": [1, 2], "web_name": ["Star", "Keeper"], "team_name": ["A", "B"],
                         "position": ["MID", "GK"], "price": [8.0, 4.5], "xp": [6.1, 3.2],
                         "role": ["XI", "XI"], "captain": [True, False], "vice": [False, True]})
    moved = pd.DataFrame({"id": [9], "web_name": ["Gone"], "team_name": ["C"], "position": ["MID"],
                          "price": [7.5]})
    return NS(gw=9, sold=moved, bought=team.iloc[[0]], team=team, free_transfers=1, hits=0)


PLAYERS = pd.DataFrame({"id": [1, 2], "status": ["d", "a"],
                        "news": ["Knock - 75% chance of playing", None]})


def test_brief_lists_the_moves_captain_and_fpl_flags():
    text = advisor.brief(_week(), PLAYERS, deadline="2026-10-31 11:00", baseline_xp=250.0,
                         plan_xp=256.5, hits_cost=0)
    assert "Gameweek 9 (deadline 2026-10-31 11:00 UTC)" in text
    assert "Transfers OUT: Gone (C, MID, £7.5m)" in text and "Transfers IN: Star" in text
    assert "Captain: Star. Vice-captain: Keeper." in text
    assert "status d; news: Knock - 75% chance of playing" in text
    assert "256.5 with this plan" in text and "250.0 making no transfers" in text


def test_advice_is_read_from_the_submit_call_and_the_request_is_well_formed():
    client = FakeClient(_response("tool_use", _search_result("https://a.example", "https://b.example"),
                                  PROSE, SUBMIT, searches=2))
    a = advisor.advise("brief", client=client)
    assert a.ok and a.verdict == "act" and a.action_score == 100      # clamped to 0-100
    assert a.searches == 2 and [s["url"] for s in a.sources] == ["https://a.example", "https://b.example"]
    assert a.player_flags[0]["source"] == "https://example.com/news"

    req = client.calls[0]
    assert req["model"] == config.ADVISOR_MODEL and req["output_config"] == {"effort": config.ADVISOR_EFFORT}
    assert req["fallbacks"] == "default" and advisor.FALLBACK_BETA in req["betas"]
    search, submit = req["tools"]
    assert search["type"] == "web_search_20260209" and search["max_uses"] == config.ADVISOR_MAX_SEARCHES
    assert submit["name"] == advisor.SUBMIT and submit["strict"] is True
    assert req["tool_choice"] == {"type": "auto"}           # forced tool choice isn't allowed


def test_a_paused_search_turn_is_resumed():
    paused = _response("pause_turn", _search_result("https://a.example"))
    client = FakeClient(paused, _response("tool_use", SUBMIT))
    a = advisor.advise("brief", client=client)
    assert a.ok and len(client.calls) == 2
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"] and second[1]["content"] is paused.content


def test_a_prose_answer_gets_one_nudge_then_gives_up():
    client = FakeClient(_response("end_turn", PROSE), _response("tool_use", SUBMIT))
    assert advisor.advise("brief", client=client).ok
    assert client.calls[1]["messages"][-1]["content"].startswith("Now call submit_advice")

    stubborn = FakeClient(_response("end_turn", PROSE), _response("end_turn", PROSE))
    a = advisor.advise("brief", client=stubborn)
    assert not a.ok and "end_turn" in a.error and len(stubborn.calls) == 2


def test_refusals_and_api_trouble_never_crash_the_run():
    assert "declined" in advisor.advise("b", client=FakeClient(_response("refusal"))).error
    down = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    assert "network" in advisor.advise("b", client=FakeClient(error=down)).error


def test_tokens_add_up_across_requests_and_cost_includes_searches():
    paused = _response("pause_turn", _search_result("https://a.example"), searches=1, tokens=(1000, 200))
    client = FakeClient(paused, _response("tool_use", SUBMIT, searches=1, tokens=(3000, 800)))
    a = advisor.advise("brief", client=client)
    assert (a.input_tokens, a.output_tokens, a.searches) == (4000, 1000, 2)
    price_in, price_out = advisor.PRICES[config.ADVISOR_MODEL]
    assert a.cost == pytest.approx((4000 * price_in + 1000 * price_out) / 1e6 + 2 * advisor.SEARCH_PRICE)
    assert advisor.approx_cost("claude-sonnet-5-5", 1_000_000, 100_000, searches=2) == pytest.approx(3.02)
    assert advisor.approx_cost("some-unpriced-model", 10, 10) is None


def test_an_empty_credit_balance_is_reported_in_plain_words():
    msg = "Your credit balance is too low to access the Anthropic API."
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    broke = anthropic.BadRequestError(
        "Error code: 400 - {...}", response=httpx.Response(400, request=req),
        body={"type": "error", "error": {"type": "invalid_request_error", "message": msg}})
    a = advisor.advise("b", client=FakeClient(error=broke))
    assert not a.ok and a.error == f"Anthropic API error 400: {msg}"


def test_a_successful_check_is_saved_and_reused_only_for_its_gameweek(tmp_path):
    a = advisor.advise("brief", client=FakeClient(_response("tool_use", SUBMIT)))
    assert a.checked_at                                       # stamped when it ran
    a.gameweek = 9
    path = tmp_path / "last_advice.json"
    advisor.save(a, path)
    again = advisor.load(path, 9)
    assert again is not None and again.action_score == a.action_score and again.player_flags == a.player_flags
    assert advisor.load(path, 10) is None                     # a new gameweek needs a new check
    assert advisor.load(tmp_path / "missing.json", 9) is None
    path.write_text("not json")
    assert advisor.load(path, 9) is None


def test_brief_names_a_chip_played_this_week():
    week = _week()
    week.chip = "bench_boost"
    assert "Chip: the plan plays the bench boost this gameweek." in advisor.brief(week, PLAYERS)
    assert "Chip:" not in advisor.brief(_week(), PLAYERS)
