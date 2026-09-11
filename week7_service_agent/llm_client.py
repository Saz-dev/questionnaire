"""
Shared NVIDIA NIM client + pricing, so agent.py and workflow.py bill
tokens the same way and the race numbers are comparable.

Reuses the same AI_API_KEY / AI_MODEL that questionnaire/rag.py reads
from .env — switched from Groq to NVIDIA's OpenAI-compatible endpoint
(integrate.api.nvidia.com) mid-session after the Groq key's daily token
quota was exhausted by this task's own testing.
"""

import os
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

    for attempt in range(max_retries):
        try:
            response = get_client().chat.completions.create(**kwargs)
            break
        except (openai.InternalServerError, openai.APIConnectionError, openai.RateLimitError):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    message = response.choices[0].message
    usage = response.usage
    usage_dict = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }
    return message, usage_dict
