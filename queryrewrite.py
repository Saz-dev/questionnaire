
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

AI_API_KEY = os.environ.get("AI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-oss-120b")
AI_BASE_URL = "https://api.groq.com/openai/v1"

# Conversational filler that adds no retrieval value.
_FILLER_RE = re.compile(
    r"\b(please|can you|could you|would you|tell me|i want to know|what about"
    r"|how about|hey|hi|hello|actually|just|kindly|do you know|is there any"
    r"|does the|what is|what are|what's|the recommended|for the|for a)\b",
    re.IGNORECASE,
)

# Domain shorthand -> manual vocabulary (the manual spells "tyre", "kms").
_ALIASES = {
    "tyres": "tyres",
    "tires": "tyres",
    "tire": "tyre",
    "kms": "km",
    "kms.": "km",
    "specs": "specifications",
    "oil": "engine oil",
    "drain plug": "drain bolt",
}


def rewrite_rule_based(question: str) -> str:
    """
    Deterministic rewrite: normalise and strip, but PRESERVE exact terms.
    The regex is case-insensitive and the final join keeps token order so
    BM25 still sees e.g. "30 A" and "1.5 mm" intact.
    """
    q = question.lower()
    for src, dst in _ALIASES.items():
        q = q.replace(src, dst)
    q = _FILLER_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or question


def _chat(messages: list[dict], retries: int = 8) -> str | None:
    """One-shot chat completion with retry on rate limits; None if no key."""
    if not AI_API_KEY:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=AI_MODEL, messages=messages, temperature=0.0
            )
            return resp.choices[0].message.content
        except Exception as exc:
            # 429 rate limit / 503 overload: wait and retry, the eval runs
            # dozens of calls back-to-back on a free-tier key.
            is_rate = getattr(exc, "status_code", None) in (429, 503)
            if not is_rate or attempt == retries - 1:
                if attempt == retries - 1:
                    print(f"  ! LLM call failed after {retries} tries: {exc}")
                return None
            time.sleep(min(30 * (2 ** attempt), 240))
    return None


def rewrite_llm(question: str) -> str:
    """
    LLM rewrite: turn the conversational question into a compact,
    keyword-rich search query. Falls back to the rule-based version when
    there is no API key.
    """
    rewritten = _chat(
        [
            {
                "role": "system",
                "content": "Rewrite this question into a short search query "
                "for an owner's manual. Keep exact codes, numbers and units "
                "like '30 A', '1.5 mm', 'MR6K-9' EXACTLY as written. Output "
                "only the query, no explanation.",
            },
            {"role": "user", "content": question},
        ]
    )
    if rewritten is None:
        return rewrite_rule_based(question)
    return rewritten.strip()


def hyde(question: str) -> str:
    """
    Hypothetical Document Embeddings: generate a passage that WOULD answer
    the question, then search with that passage's embedding.

    Returns the passage (the caller embeds it). Without an API key it
    returns the question unchanged — HyDE needs a generative model.
    """
    passage = _chat(
        [
            {
                "role": "system",
                "content": "You are writing retrieval training data. Write a "
                "2-3 sentence factual passage from a motorcycle owner's manual "
                "that would answer the user's question. Use exact numbers, "
                "codes and units. Do not invent facts outside the manual's "
                "typical content (oil, tyres, brakes, chain, spark plug, "
                "fuses, warranty, torque values).",
            },
            {"role": "user", "content": question},
        ]
    )
    return passage.strip() if passage else question


if __name__ == "__main__":
    qs = [
        "please tell me how much air do I put in my tyres, mate?",
        "what is the main fuse rating?",
        "specs of the drive chain?",
    ]
    for q in qs:
        print(f"IN : {q}")
        print(f"OUT: {rewrite_rule_based(q)}")
        print()
