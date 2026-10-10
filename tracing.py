import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import obs
import rag as rag_module

TRACE_PATH = Path("traces.jsonl")

# PII redaction lives in obs.py (single implementation; runs BEFORE any write).
redact_pii = obs.redact_pii


def make_trace(question: str, result: dict, mode: str, temperature: float = 0.2,
               user_id: str | None = None, session_id: str | None = None) -> dict:
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

    # Per-span latency / tokens / cost (retrieval, generation). `timing_ms`
    # and `usage` are present on RAG.answer() results; the streaming endpoint
    # passes them through the same keys.
    timing = result.get("timing_ms") or {}
    usage = result.get("usage") or {}
    tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    rec = obs.TraceRecorder()
    rec.add_span("retrieval", "retrieval", timing.get("retrieval", 0.0), cost_usd=obs.RETRIEVAL_COST_USD,
                 mode=mode, candidates=len(pool))
    if result.get("generation_method") != "refused" and result["grounded"]:
        rec.add_span("generation", "generation", timing.get("generation", 0.0), tokens_in=tin,
                     tokens_out=tout, cost_usd=obs.llm_cost(rag_module.AI_MODEL, tin, tout))

    trace = obs.build_trace(
        question=question, answer=result["answer"], prompt_version=rag_module.PROMPT_VERSION,
        model=rag_module.AI_MODEL, spans=rec.spans,
        retrieved_context_ids=[h.get("chunk_id") for h in pool], source="rag",
        user_id=user_id, session_id=session_id, mode=mode,
        extra={
            "temperature": temperature,
            "retrieved": retrieved,
            "evidence_chunk_ids": [h.get("chunk_id") for h in result["evidence"]],
            "grounded": result["grounded"],
        },
    )
    return trace


def write_trace(trace: dict, path: Path = TRACE_PATH) -> None:
    obs.write_trace_v2(trace, path)


def load_traces(path: Path = TRACE_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def find_trace(trace_id: str, path: Path = TRACE_PATH) -> dict | None:
    for t in load_traces(path):
        if t["trace_id"] == trace_id:
            return t
    return None
