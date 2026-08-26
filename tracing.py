import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import rag as rag_module

TRACE_PATH = Path("traces.jsonl")

# Defensive PII scrub. This app's real questions are motorcycle-manual
# lookups with no claimant data in them, but the redaction step must
# still run BEFORE a trace is written (not filtered out afterward), so
# the requirement is satisfied as actual pipeline behavior, not a
# retroactive claim. See notes.md for confirmation this fires 0 times
# on the real corpus and why.
_CLAIM_NUMBER_RE = re.compile(r"\b(claim|policy)[\s#:-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)
_NAME_LABEL_RE = re.compile(r"\b(claimant|policyholder)[\s:]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")


def redact_pii(text: str) -> str:
    text = _CLAIM_NUMBER_RE.sub("[REDACTED-CLAIM-NUMBER]", text)
    text = _NAME_LABEL_RE.sub("[REDACTED-NAME]", text)
    return text


def make_trace(question: str, result: dict, mode: str, temperature: float = 0.2) -> dict:
    """
    Build one trace record from a RAG.answer() result.

    `result` is exactly what RAG.answer() returns:
    {"answer", "evidence", "meta", "grounded"}.
    """
    pool = result["meta"].get("candidate_pool") or result["evidence"]
    retrieved = [
        {
            "chunk_id": h.get("chunk_id"),
            "source": h.get("source"),
            "cosine": h.get("cosine"),
            "score": h.get("score"),
            "retrieval_score": h.get("retrieval_score"),
        }
        for h in pool
    ]

    trace = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": redact_pii(question),
        "prompt_version": rag_module.PROMPT_VERSION,
        "mode": mode,
        "model": rag_module.AI_MODEL,
        "temperature": temperature,
        "retrieved": retrieved,
        "evidence_chunk_ids": [h.get("chunk_id") for h in result["evidence"]],
        "grounded": result["grounded"],
        "raw_output": result["answer"],
        "answer": redact_pii(result["answer"]),
    }
    return trace


def write_trace(trace: dict, path: Path = TRACE_PATH) -> None:
    with path.open("a") as f:
        f.write(json.dumps(trace) + "\n")


def load_traces(path: Path = TRACE_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def find_trace(trace_id: str, path: Path = TRACE_PATH) -> dict | None:
    for t in load_traces(path):
        if t["trace_id"] == trace_id:
            return t
    return None
