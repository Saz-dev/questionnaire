"""
Shared NVIDIA NIM client + pricing, so agent.py and workflow.py bill
tokens the same way and the race numbers are comparable.

Reuses the same AI_API_KEY / AI_MODEL that questionnaire/rag.py reads
from .env — switched from Groq to NVIDIA's OpenAI-compatible endpoint
(integrate.api.nvidia.com) mid-session after the Groq key's daily token
quota was exhausted by this task's own testing.
"""

import os
import re
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

AI_API_KEY = os.environ.get("AI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "gpt-4o-mini")
AI_BASE_URL = os.environ.get("AI_BASE_URL", "https://api.openai.com/v1")

# NVIDIA doesn't publish integrate.api.nvidia.com pricing directly; this
# uses OpenRouter's published rate for the same model (nemotron-3-super-
# 120b-a12b) as the best available proxy — see WEEK7_REPORT.md. $ per
# token, not per million — pre-divided so cost math elsewhere is a plain
# multiply.
PRICE_PER_INPUT_TOKEN = 0.085 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 0.40 / 1_000_000

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not AI_API_KEY:
            raise RuntimeError(
                "AI_API_KEY not set — copy questionnaire/.env's key or set "
                "your own before running the race."
            )
        _client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
    return _client


def cost_of(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * PRICE_PER_INPUT_TOKEN + completion_tokens * PRICE_PER_OUTPUT_TOKEN


def chat(messages: list[dict], tools: list[dict] | None = None, tool_choice: str | None = None, max_retries: int = 4):
    """
    One chat-completion call. Returns (message, usage_dict).

    NVIDIA's shared integrate.api.nvidia.com endpoint intermittently
    returns transient 5xx errors under load — retried with backoff so a
    single flaky call doesn't crash a 10-ticket race.
    """
    kwargs = {"model": AI_MODEL, "messages": messages, "temperature": 0}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    rate_limit_wait_s = 0.0
    for attempt in range(max_retries):
        try:
            response = get_client().chat.completions.create(**kwargs)
            break
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            # Groq's daily-token-quota 429s name a wait ("Please try again
            # in 13m35.184s") that dwarfs the transient-5xx backoff below —
            # parse it so a long eval run (trajectory_eval.py: 11 tickets x
            # 2 conditions) survives instead of burning its retry budget in
            # a few seconds and crashing partway through. Capped well under
            # agent.py's own max_wall_clock_s so one rate-limit wait can't
            # look like the AGENT looping forever — see rate_limit_wait_s
            # below, which agent.py adds back onto its own clock.
            wait_s = 5.0
            match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
            if match:
                minutes, seconds = match.groups()
                wait_s = (int(minutes) * 60 if minutes else 0) + float(seconds) + 2
            wait_s = min(wait_s, 900)
            rate_limit_wait_s += wait_s
            time.sleep(wait_s)
        except (openai.InternalServerError, openai.APIConnectionError):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    message = response.choices[0].message
    usage = response.usage
    usage_dict = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        # Provider-side dead time, not the agent's own thinking time — the
        # caller's wall-clock budget should not count this against it.
        "rate_limit_wait_s": rate_limit_wait_s,
    }
    return message, usage_dict
