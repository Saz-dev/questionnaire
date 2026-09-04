"""
Week 6 — LLM-as-judge for the one criterion assertions.py can't check: does the
answer stay within what the retrieved evidence actually supports?

Same Groq/OpenAI-compatible client pattern as rag.py/queryrewrite.py.

Run:
    python judge.py v1                    # score all 25 judge-eligible cases with judge_v1.txt
    python judge.py v2                    # ... with judge_v2.txt
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

AI_API_KEY = os.environ.get("AI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-oss-120b")
AI_BASE_URL = "https://api.groq.com/openai/v1"

EVAL_SET_PATH = Path("eval_set.jsonl")
CHUNKS_CACHE_PATH = Path("qdrant_storage/chunks.json")

VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)
REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)


def _load_chunk_text_by_id() -> dict[str, str]:
    chunks = json.loads(CHUNKS_CACHE_PATH.read_text())
    out = {}
    for i, c in enumerate(chunks):
        cid = f"{c['source']}::{c.get('chunk_id', i)}"
        out[cid] = c["text"]
    return out


def _build_context(evidence_chunk_ids: list[str], chunk_text_by_id: dict[str, str]) -> str:
    parts = []
    for cid in evidence_chunk_ids:
        text = chunk_text_by_id.get(cid, "<chunk text unavailable>")
        source = cid.split("::")[0]
        parts.append(f"[{source}]\n{text}")
    return "\n\n".join(parts)


def _call_llm(prompt: str, retries: int = 8) -> str | None:
    if not AI_API_KEY:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            is_rate = getattr(exc, "status_code", None) in (429, 503)
            if not is_rate or attempt == retries - 1:
                print(f"  ! judge LLM call failed: {exc}")
                return None
            time.sleep(min(30 * (2 ** attempt), 240))
    return None


def judge_one(template: str, case: dict, chunk_text_by_id: dict[str, str]) -> dict:
    context = _build_context(case.get("evidence_chunk_ids", []), chunk_text_by_id)
    prompt = template.format(context=context, question=case["question"], answer=case["answer"])
    raw = _call_llm(prompt)
    if raw is None:
        return {"verdict": None, "reason": "LLM call failed / no API key", "raw": None}

    v_match = VERDICT_RE.search(raw)
    r_match = REASON_RE.search(raw)
    verdict = v_match.group(1).upper() if v_match else None
    reason = r_match.group(1).strip() if r_match else raw.strip()
    return {"verdict": verdict, "reason": reason, "raw": raw}


def run(version: str) -> list[dict]:
    template_path = Path(f"judge_{version}.txt")
    template = template_path.read_text()
    chunk_text_by_id = _load_chunk_text_by_id()

    results = []
    with EVAL_SET_PATH.open() as f:
        for line in f:
            case = json.loads(line)
            if not case.get("judge_eligible"):
                continue
            result = judge_one(template, case, chunk_text_by_id)
            result["id"] = case["id"]
            result["question"] = case["question"]
            results.append(result)
            print(f"{case['id']:8s} {result['verdict'] or 'ERROR':6s} {result['reason'][:80]}")
    return results


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    results = run(version)
    out_path = Path(f"judge_results_{version}.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(results)} verdicts to {out_path}")
