"""
LLM helper for the Week-9 MCP agent. Same env contract and retry
behaviour as week7_service_agent/llm_client.py and rag.py: reads
AI_API_KEY / AI_MODEL / AI_BASE_URL from questionnaire/.env, talks to an
OpenAI-compatible endpoint (Groq by default).

The model lives HERE, in the host — never inside any MCP server. See
agent.py:MODEL_CALL_LOCATION for the one-line statement.
"""

import os
import time

import openai
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

AI_API_KEY = os.environ.get("AI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-oss-120b")
AI_BASE_URL = os.environ.get("AI_BASE_URL", "https://api.groq.com/openai/v1")


def chat(messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", max_retries: int = 4):
    """One chat-completion call. Returns (message, usage). Retries 5xx/429."""
    if not AI_API_KEY:
        raise RuntimeError("AI_API_KEY not set — copy questionnaire/.env's key or set your own.")
    client = openai.OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
    kwargs = {"model": AI_MODEL, "messages": messages, "temperature": 0}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    message = response.choices[0].message
    usage = response.usage
    return message, {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }