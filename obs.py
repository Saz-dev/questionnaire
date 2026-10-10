"""
Week 11 — observability layer (trace schema v2 + searchable index).

One place that knows how to:
  * redact PII *before* anything is written (JSONL and SQLite alike),
  * price a request per stage (retrieval / generation / tools),
  * classify the INPUT (input_type) and the OUTPUT (says_covered, cites_exclusion)
    at write time, so "the answer said it was covered" is an indexed field and
    not a grep,
  * write every trace to traces.jsonl (append-only source of truth) AND to
    traces.db (SQLite + FTS5 index, rebuildable from the JSONL).

Both the RAG path (tracing.py) and the Week-9 agent path (week9_mcp/agent.py)
emit the same v2 record, so there is exactly one log to search.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
TRACE_PATH = HERE / "traces.jsonl"
DB_PATH = HERE / "traces.db"

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Cost model. $ per token (pre-divided). Groq's published rate for
# openai/gpt-oss-120b: $0.15 / 1M input, $0.75 / 1M output. Retrieval is local
# (MiniLM + cross-encoder on CPU) and MCP tools are local stdio subprocesses, so
# their marginal $ is 0 — recorded as 0.0 on purpose, not omitted, so the stage
# split is explicit. Change here, nowhere else.
# ---------------------------------------------------------------------------
PRICES = {
    "openai/gpt-oss-120b": {"in": 0.15 / 1_000_000, "out": 0.75 / 1_000_000},
}
DEFAULT_PRICE = PRICES["openai/gpt-oss-120b"]
RETRIEVAL_COST_USD = 0.0
TOOL_CALL_COST_USD = 0.0

# Groq's published on-demand limits for openai/gpt-oss-120b (used by tenx.md).
RATE_LIMIT_TPM = 8_000
RATE_LIMIT_RPM = 30


def llm_cost(model: str | None, tokens_in: int, tokens_out: int) -> float:
    p = PRICES.get(model or "", DEFAULT_PRICE)
    return tokens_in * p["in"] + tokens_out * p["out"]


# ---------------------------------------------------------------------------
# PII redaction — runs on write. Free-text adjuster notes are never stored
# verbatim: only a hash + length survive.
# ---------------------------------------------------------------------------
_CLAIM_NUMBER_RE = re.compile(r"\b(claim|policy)[\s#:-]*[A-Z0-9-]{4,}\b", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"\b(?:CLM|POL)-\d{4}-\d{5}\b")
_NAME_LABEL_RE = re.compile(r"\b(claimant|policyholder)[\s:]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
_INITIAL_NAME_RE = re.compile(r"\b[A-Z]\.\s[A-Z][a-z]{2,}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s.-]?){9,13}\d(?!\d)")
_NOTE_RE = re.compile(r"(adjuster(?:'s)?\s+note[s]?\s*[:\-]\s*)(.+)", re.IGNORECASE | re.DOTALL)


def _note_stub(m: re.Match) -> str:
    body = m.group(2)
    digest = hashlib.sha256(body.encode()).hexdigest()[:10]
    return f"{m.group(1)}[REDACTED-NOTE sha256={digest} len={len(body)}]"


def redact_pii(text: str | None) -> str | None:
    if text is None:
        return None
    text = _NOTE_RE.sub(_note_stub, text)
    text = _CLAIM_NUMBER_RE.sub("[REDACTED-CLAIM-NUMBER]", text)
    text = _BARE_ID_RE.sub("[REDACTED-ID]", text)
    text = _NAME_LABEL_RE.sub("[REDACTED-NAME]", text)
    text = _INITIAL_NAME_RE.sub("[REDACTED-NAME]", text)
    text = _EMAIL_RE.sub("[REDACTED-EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED-PHONE]", text)
    return text


# ---------------------------------------------------------------------------
# Input + output classification (rule-based, deterministic, computed on write)
# ---------------------------------------------------------------------------
_COVERAGE_Q = re.compile(r"\b(cover(?:ed|age)?|exclu\w+|insured|payable|eligible|entitled)\b", re.I)
_STATUS_Q = re.compile(r"\b(status|denied|approved|adjuster|notes?|why was)\b", re.I)
_LIMIT_Q = re.compile(r"\b(limit|deductible|cap|amount|how much|premium)\b", re.I)


def classify_input_type(question: str) -> str:
    if _COVERAGE_Q.search(question):
        return "coverage"
    if _STATUS_Q.search(question):
        return "claim_status"
    if _LIMIT_Q.search(question):
        return "limit"
    return "other"


_NEG_COVERED = re.compile(
    r"\b(not|isn['’]?t|aren['’]?t|n['’]?t|never|un)[\s‑‐-]*(be\s+)?(covered|payable)"
    r"|^\W*no\b|\bno\b[^.\n]{0,40}\bcover|\bexclud\w+|\bdenied\b|\bnot covered\b",
    re.I,
)
_POS_COVERED = re.compile(r"\bcovered\b|\bpayable\b|^\W*yes\b|\bprovides?\b[^.\n]{0,30}\bcoverage\b", re.I)
_EXCLUSION_TERMS = re.compile(r"\bexclu\w+|\bnot covered\b|\bdenied\b|\bdoes not cover\b", re.I)


def output_flags(answer: str | None) -> dict:
    """says_covered: the answer asserts coverage and nowhere negates it.
    says_not_covered: the answer negates/excludes. cites_exclusion: it names an exclusion."""
    a = (answer or "")
    neg = bool(_NEG_COVERED.search(a))
    pos = bool(_POS_COVERED.search(a))
    return {
        "says_covered": pos and not neg,
        "says_not_covered": neg,
        "cites_exclusion": bool(_EXCLUSION_TERMS.search(a)),
    }


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------
class TraceRecorder:
    """Collects spans for one request.  Use `with rec.span("retrieval", "retrieval"):`
    for timed sections or `rec.add_span(...)` when duration/tokens are already known."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.start_ts = datetime.now(timezone.utc)
        self.spans: list[dict] = []

    def add_span(self, name: str, kind: str, duration_ms: float, tokens_in: int = 0,
                 tokens_out: int = 0, cost_usd: float = 0.0, **attrs) -> dict:
        s = {
            "name": name, "kind": kind,
            "start_ms": round((time.perf_counter() - self.t0) * 1000 - duration_ms, 1),
            "duration_ms": round(duration_ms, 1),
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 8),
            "attrs": attrs,
        }
        self.spans.append(s)
        return s

    @contextmanager
    def span(self, name: str, kind: str, **attrs):
        t = time.perf_counter()
        holder: dict = {}
        try:
            yield holder
        finally:
            self.add_span(
                name, kind, (time.perf_counter() - t) * 1000,
                tokens_in=holder.get("tokens_in", 0), tokens_out=holder.get("tokens_out", 0),
                cost_usd=holder.get("cost_usd", 0.0), **attrs, **holder.get("attrs", {}),
            )


def totals(spans: list[dict]) -> dict:
    by_stage = {"retrieval": 0.0, "generation": 0.0, "tools": 0.0}
    for s in spans:
        by_stage[s["kind"]] = by_stage.get(s["kind"], 0.0) + s["cost_usd"]
    return {
        "latency_ms": round(sum(s["duration_ms"] for s in spans), 1),
        "tokens_in": sum(s["tokens_in"] for s in spans),
        "tokens_out": sum(s["tokens_out"] for s in spans),
        "cost_usd": round(sum(s["cost_usd"] for s in spans), 8),
        "cost_by_stage": {k: round(v, 8) for k, v in by_stage.items()},
    }


def build_trace(*, question: str, answer: str, prompt_version: str, model: str | None,
                spans: list[dict], retrieved_context_ids: list[str], source: str,
                user_id: str | None = None, session_id: str | None = None,
                mode: str | None = None, tools_called: list[str] | None = None,
                extra: dict | None = None, error: str | None = None,
                ts: str | None = None, trace_id: str | None = None) -> dict:
    flags = output_flags(answer)
    trace = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id or str(uuid.uuid4()),
        "timestamp": ts or datetime.now(timezone.utc).isoformat(),
        "source": source,                      # "rag" | "agent"
        "user_id": user_id or "anonymous",
        "session_id": session_id,
        "input_type": classify_input_type(question),
        "prompt_version": prompt_version,
        "model": model,
        "mode": mode,
        "question": redact_pii(question),
        "answer": redact_pii(answer),
        "raw_output_sha256": hashlib.sha256((answer or "").encode()).hexdigest()[:16],
        "retrieved_context_ids": retrieved_context_ids,
        "tools_called": tools_called or [],
        "spans": spans,
        "totals": totals(spans),
        **flags,
        "error": error,
    }
    if extra:
        trace.update(extra)
    return trace


# ---------------------------------------------------------------------------
# Storage: JSONL (truth) + SQLite/FTS5 (index)
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS traces(
  trace_id TEXT PRIMARY KEY, ts TEXT, source TEXT, user_id TEXT, session_id TEXT,
  prompt_version TEXT, input_type TEXT, model TEXT,
  says_covered INTEGER, says_not_covered INTEGER, cites_exclusion INTEGER,
  cost_usd REAL, latency_ms REAL, tokens_in INTEGER, tokens_out INTEGER,
  question TEXT, answer TEXT, json TEXT
);
CREATE INDEX IF NOT EXISTS ix_ts ON traces(ts);
CREATE INDEX IF NOT EXISTS ix_user ON traces(user_id);
CREATE INDEX IF NOT EXISTS ix_pv ON traces(prompt_version);
CREATE INDEX IF NOT EXISTS ix_it ON traces(input_type);
CREATE INDEX IF NOT EXISTS ix_cov ON traces(says_covered, cites_exclusion);
CREATE INDEX IF NOT EXISTS ix_cost ON traces(cost_usd);
CREATE VIRTUAL TABLE IF NOT EXISTS traces_fts USING fts5(trace_id UNINDEXED, question, answer);
"""


def _db(path: Path = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(_DDL)
    return con


def normalize_legacy(t: dict) -> dict:
    """Lift a pre-v2 row (RAG path, schema v1) to v2 fields, additively."""
    if t.get("schema_version") == SCHEMA_VERSION:
        return t
    t = dict(t)
    answer = t.get("answer") or ""
    t.setdefault("schema_version", 1)
    t.setdefault("source", "rag")
    t.setdefault("user_id", "anonymous")
    t.setdefault("session_id", None)
    t.setdefault("input_type", classify_input_type(t.get("question", "")))
    t.setdefault("retrieved_context_ids", t.get("evidence_chunk_ids") or [])
    t.setdefault("tools_called", [])
    t.setdefault("spans", [])
    t.setdefault("totals", {"latency_ms": 0.0, "tokens_in": 0, "tokens_out": 0,
                            "cost_usd": 0.0, "cost_by_stage": {"retrieval": 0.0, "generation": 0.0, "tools": 0.0}})
    for k, v in output_flags(answer).items():
        t.setdefault(k, v)
    return t


def index_trace(t: dict, con: sqlite3.Connection | None = None) -> None:
    own = con is None
    con = con or _db()
    t = normalize_legacy(t)
    tot = t["totals"]
    con.execute(
        "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (t["trace_id"], t["timestamp"], t.get("source"), t.get("user_id"), t.get("session_id"),
         t.get("prompt_version"), t.get("input_type"), t.get("model"),
         int(t["says_covered"]), int(t["says_not_covered"]), int(t["cites_exclusion"]),
         tot["cost_usd"], tot["latency_ms"], tot["tokens_in"], tot["tokens_out"],
         t.get("question"), t.get("answer"), json.dumps(t)),
    )
    con.execute("DELETE FROM traces_fts WHERE trace_id=?", (t["trace_id"],))
    con.execute("INSERT INTO traces_fts VALUES (?,?,?)", (t["trace_id"], t.get("question"), t.get("answer")))
    if own:
        con.commit()
        con.close()


def write_trace_v2(trace: dict, path: Path = TRACE_PATH) -> dict:
    """Append to JSONL and index. Redaction already happened in build_trace."""
    with path.open("a") as f:
        f.write(json.dumps(trace) + "\n")
    index_trace(trace)
    return trace


def backfill(path: Path = TRACE_PATH) -> int:
    """Rebuild traces.db from traces.jsonl (old v1 rows are normalized, not rewritten)."""
    con = _db()
    n = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                t.update(output_flags(t.get("answer")))   # flags are derived: recompute on rebuild
                index_trace(t, con)
                n += 1
    con.commit()
    con.close()
    return n


# ---------------------------------------------------------------------------
# Query helpers (used by drill_tools.py, cost_report.py and the API)
# ---------------------------------------------------------------------------
def search_traces(*, since: str | None = None, until: str | None = None, user_id: str | None = None,
                  prompt_version: str | None = None, input_type: str | None = None,
                  says_covered: bool | None = None, cites_exclusion: bool | None = None,
                  text: str | None = None, min_cost: float | None = None,
                  source: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    where, args = [], []
    if since:
        where.append("t.ts >= ?"); args.append(since)
    if until:
        where.append("t.ts < ?"); args.append(until)
    for col, val in (("user_id", user_id), ("prompt_version", prompt_version),
                     ("input_type", input_type), ("source", source)):
        if val:
            where.append(f"t.{col} = ?"); args.append(val)
    if says_covered is not None:
        where.append("t.says_covered = ?"); args.append(int(says_covered))
    if cites_exclusion is not None:
        where.append("t.cites_exclusion = ?"); args.append(int(cites_exclusion))
    if min_cost is not None:
        where.append("t.cost_usd >= ?"); args.append(min_cost)
    join = ""
    if text:
        join = "JOIN traces_fts f ON f.trace_id = t.trace_id"
        where.append("traces_fts MATCH ?"); args.append(text)
    sql = f"SELECT t.* FROM traces t {join} " + ("WHERE " + " AND ".join(where) if where else "") + \
          " ORDER BY t.ts DESC LIMIT ? OFFSET ?"
    con = _db()
    rows = [dict(r) for r in con.execute(sql, (*args, limit, offset))]
    con.close()
    return rows


def get_trace(trace_id: str) -> dict | None:
    con = _db()
    r = con.execute("SELECT json FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
    con.close()
    return json.loads(r["json"]) if r else None
