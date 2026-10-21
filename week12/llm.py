"""Minimal OpenAI-compatible chat client (stdlib only; Groq by default, same env contract
as week9_mcp/llm_client.py). Used ONLY for (a) the optional LLMPlanner and (b) the optional
one-sentence narrative. The deterministic path never imports a key.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _load_env():
    p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
MODEL = os.environ.get("AI_MODEL", "openai/gpt-oss-120b")
BASE_URL = os.environ.get("AI_BASE_URL", "https://api.groq.com/openai/v1")


def available() -> bool:
    return bool(os.environ.get("AI_API_KEY"))


def chat(messages: list[dict], tools: list[dict] | None = None, retries: int = 4,
         transport=None) -> tuple[dict, dict]:
    """Return (assistant_message_dict, usage). `transport` lets tests inject a fake."""
    body = {"model": MODEL, "messages": messages, "temperature": 0}
    if tools:
        body["tools"], body["tool_choice"] = tools, "auto"
    if transport is not None:
        resp = transport(body)
    else:
        key = os.environ.get("AI_API_KEY")
        if not key:
            raise RuntimeError("AI_API_KEY not set")
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503) or attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
    usage = resp.get("usage") or {}
    return resp["choices"][0]["message"], {
        "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}
