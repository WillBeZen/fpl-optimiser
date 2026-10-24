"""
check_claude.py — one tiny request to confirm your Anthropic key works.

    python scripts/check_claude.py

Sends a one-line prompt to config.ADVISOR_MODEL at low effort, with no web search,
and prints the reply, the tokens used and their approximate cost (a fraction of a
cent). Reads ANTHROPIC_API_KEY from .env; never prints it.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config
from fpl_optimiser.claude import advisor


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(config.ROOT / ".env")
    import anthropic

    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=config.ADVISOR_MODEL, max_tokens=512, output_config={"effort": "low"},
            betas=[advisor.FALLBACK_BETA], fallbacks="default",
            messages=[{"role": "user", "content": "Reply with exactly one word: OK"}])
    except anthropic.AuthenticationError:
        print("FAILED: the API key was rejected. Check ANTHROPIC_API_KEY in .env.")
        return 1
    except anthropic.PermissionDeniedError as exc:
        print(f"FAILED: the key isn't allowed to use {config.ADVISOR_MODEL}: {advisor.api_message(exc)}")
        return 1
    except anthropic.RateLimitError:
        print("FAILED: rate limited. Try again in a minute.")
        return 1
    except anthropic.APIStatusError as exc:
        print(f"FAILED: API error {exc.status_code}: {advisor.api_message(exc)}")
        return 1
    except anthropic.APIConnectionError:
        print("FAILED: couldn't reach the Anthropic API (network).")
        return 1
    except anthropic.AnthropicError as exc:
        print(f"FAILED: {exc}")
        return 1

    if response.stop_reason == "refusal":
        print("FAILED: the request was declined.")
        return 1
    text = " ".join(b.text for b in response.content if b.type == "text").strip()
    u = response.usage
    cost = advisor.approx_cost(response.model, u.input_tokens, u.output_tokens)
    print(f"OK: {response.model} replied {text!r} (stop: {response.stop_reason})")
    print(f"tokens: {u.input_tokens} in, {u.output_tokens} out"
          + (f"  ~${cost:.4f}" if cost is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
