"""
HTTP API for the RAG pipeline.

This is a thin FastAPI wrapper around the existing pipeline modules
(loader, chunker, embedder, vectordb, reranker, rag) — it does not
reimplement any retrieval/generation logic, it just exposes the real
functions over HTTP so a browser-based frontend can call them.

Run with:
    uvicorn api:app --reload --port 8000
"""

import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import json

import embedder
import rag as rag_module
import reranker
import vectordb
from chunker import chunk_documents
from loader import load_pdfs
from rag import RAG
from tracing import make_trace, write_trace
from vectordb import VectorDB, _hash_documents
from week6_api import build_week6_summary, get_label_queue, submit_label

load_dotenv()

DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file — basic upload hygiene

RETRIEVAL_MODES = ("semantic", "hybrid", "rerank", "hybrid_rerank", "mmr", "hybrid_mmr", "rewrite", "hyde")
RetrievalMode = Literal["semantic", "hybrid", "rerank", "hybrid_rerank", "mmr", "hybrid_mmr", "rewrite", "hyde"]

MODE_INFO = [
    {"id": "semantic", "label": "Semantic", "family": "dense",
     "description": "Dense vector cosine similarity search over all-MiniLM-L6-v2 embeddings."},
    {"id": "hybrid", "label": "Hybrid (BM25 + RRF)", "family": "fusion",
     "description": "Fuses semantic search with Okapi BM25 keyword search via Reciprocal Rank Fusion."},
    {"id": "rerank", "label": "Semantic + Rerank", "family": "rerank",
     "description": "Semantic search over a larger candidate pool, reordered by a cross-encoder. Default mode."},
    {"id": "hybrid_rerank", "label": "Hybrid + Rerank", "family": "rerank",
     "description": "Hybrid (BM25+RRF) candidates, reordered by a cross-encoder."},
    {"id": "mmr", "label": "MMR (diversity)", "family": "diversity",
     "description": "Maximal Marginal Relevance over semantic candidates — trades some relevance for less redundancy."},
    {"id": "hybrid_mmr", "label": "Hybrid + MMR", "family": "diversity",
     "description": "MMR diversity selection applied to the fused hybrid candidate pool."},
    {"id": "rewrite", "label": "Query Rewrite", "family": "query-transform",
     "description": "Rewrites the question into a compact, keyword-rich query (LLM, falls back to rules) before semantic search."},
    {"id": "hyde", "label": "HyDE", "family": "query-transform",
     "description": "Hypothetical Document Embeddings — embeds an LLM-generated hypothetical answer passage instead of the question."},
]

PIPELINE_STAGES = [
    {"id": "query", "label": "Query"},
    {"id": "embedding", "label": "Embedding"},
    {"id": "retrieval", "label": "Retrieval"},
    {"id": "ranking", "label": "Ranking"},
    {"id": "context", "label": "Context assembly"},
    {"id": "generation", "label": "Generation"},
]


def ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ----------------------------------------------------------------------
# Application state — one VectorDB/RAG instance, built once (or loaded
# from the on-disk cache) and reused across requests.
# ----------------------------------------------------------------------

class _State:
    def __init__(self):
        self.db: Optional[VectorDB] = None


state = _State()


def get_db() -> VectorDB:
    if state.db is None:
        state.db = vectordb.build_index(DOCUMENTS_DIR)
    return state.db


def get_rag() -> RAG:
    return RAG(get_db())


def _read_manifest_fingerprint() -> Optional[str]:
    if not vectordb.MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(vectordb.MANIFEST_PATH.read_text()).get("fingerprint")
    except Exception:
        return None


def _index_in_sync() -> bool:
    return _read_manifest_fingerprint() == _hash_documents(DOCUMENTS_DIR)


def _rebuild_index(db: VectorDB) -> list[dict]:
    """Re-run load -> chunk -> embed -> store, returning real per-stage timings."""
    stages = []

    t0 = time.perf_counter()
    documents = load_pdfs(DOCUMENTS_DIR)
    stages.append({"stage": "parse", "duration_ms": ms(t0), "detail": {"documents": len(documents)}})

    t1 = time.perf_counter()
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    stages.append({"stage": "chunk", "duration_ms": ms(t1), "detail": {"chunks": len(chunks)}})

    t2 = time.perf_counter()
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    stages.append({"stage": "embed", "duration_ms": ms(t2), "detail": {"vectors": len(vectors)}})

    t3 = time.perf_counter()
    fingerprint = _hash_documents(DOCUMENTS_DIR)
    db.rebuild(chunks, vectors, fingerprint)
    stages.append({"stage": "index", "duration_ms": ms(t3), "detail": {"chunk_count": len(chunks)}})

    return stages


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------

class ChunkHit(BaseModel):
    text: str
    source: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    cosine: Optional[float] = None
    retrieval_score: Optional[float] = None
    mmr_score: Optional[float] = None


class RetrievalMeta(BaseModel):
    mode: str
    candidate_pool: Optional[list[ChunkHit]] = None
    rewritten: Optional[str] = None
    hypothetical: Optional[str] = None


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: RetrievalMode = "rerank"
    top_k: int = Field(3, ge=1, le=20)
    candidates: int = Field(50, ge=1, le=200)
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)


class RetrieveResponse(BaseModel):
    question: str
    mode: str
    hits: list[ChunkHit]
    meta: RetrievalMeta
    duration_ms: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: RetrievalMode = "rerank"
    top_k: int = Field(3, ge=1, le=20)
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)


class TimingMs(BaseModel):
    retrieval: float
    generation: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    evidence: list[ChunkHit]
    meta: RetrievalMeta
    best_score: float
    min_score: float
    timing_ms: TimingMs
    generation_method: Literal["llm", "fallback_synthesis", "refused"]
    prompt: Optional[str] = None


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    modified_at: str
    chunk_count: int
    status: Literal["indexed", "pending_reindex", "not_indexed"]


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int
    index_in_sync: bool


class RebuildStage(BaseModel):
    stage: str
    duration_ms: float
    detail: dict


class RebuildResponse(BaseModel):
    stages: list[RebuildStage]
    document_count: int
    chunk_count: int


class DeleteResponse(BaseModel):
    deleted: str
    rebuild: RebuildResponse


# ----------------------------------------------------------------------
# Week 6 eval — assertions + judge results (see build_eval_set.py, assertions.py,
# judge.py; this endpoint only reads what those already produced on disk)
# ----------------------------------------------------------------------

class ModeBreakdownRow(BaseModel):
    mode: str
    pass_count: int
    total: int
    rate: float


class EvalCaseDetail(BaseModel):
    id: str
    question: str
    failure_mode_tag: str
    retrieval_mode: str
    regression_case: bool
    judge_eligible: bool
    expected_grounded: bool
    assertions: dict[str, Optional[bool]]
    assertions_passed: bool
    answer: str
    hand_label: Optional[str] = None
    judge_v1_verdict: Optional[str] = None
    judge_v2_verdict: Optional[str] = None


class AgreementStat(BaseModel):
    rate: float
    matched: int
    total: int


class Disagreement(BaseModel):
    id: str
    question: str
    hand_label: str
    hand_label_reason: str
    judge_v1_verdict: str
    judge_v1_reason: str
    judge_v2_verdict: Optional[str] = None
    judge_v2_reason: Optional[str] = None
    resolved_in_v2: bool


class Week6EvalResponse(BaseModel):
    mode_breakdown: list[ModeBreakdownRow]
    overall_pass_rate: float
    overall_pass_count: int
    overall_total: int
    regression_cases: list[EvalCaseDetail]
    cases: list[EvalCaseDetail]
    assertion_names: list[str]
    replaced_judge_criteria: list[str]
    remaining_judge_criteria: list[str]
    labels_recorded_at: Optional[str] = None
    agreement_before: Optional[AgreementStat] = None
    agreement_after: Optional[AgreementStat] = None
    disagreements: list[Disagreement]
    prediction: Optional[str] = None
    prediction_outcome: Optional[str] = None
    judge_v1_prompt: Optional[str] = None
    judge_v2_prompt: Optional[str] = None


class LabelQueueItem(BaseModel):
    id: str
    question: str
    answer: str
    context: str
    existing_verdict: Optional[str] = None
    existing_reason: Optional[str] = None


class LabelSubmitRequest(BaseModel):
    id: str
    verdict: Literal["PASS", "FAIL"]
    reason: str = Field(..., min_length=1)
    labeler: str = Field("human reviewer", min_length=1)


class LabelSubmitResponse(BaseModel):
    PASS: int
    FAIL: int
    total: int
    remaining: int


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

app = FastAPI(title="RAG Lab API", version="1.0.0")

_origins_env = os.environ.get("FRONTEND_ORIGIN")
app.add_middleware(
    CORSMiddleware,
    # Vite picks the next free port (5173, 5174, ...) if the default is
    # already taken, so pin to one exact origin only when explicitly
    # configured; otherwise allow any localhost/127.0.0.1 port for local dev.
    allow_origins=[_origins_env] if _origins_env else [],
    allow_origin_regex=None if _origins_env else r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # Load the cached index (if present) so the first real request isn't
    # slowed down by BM25 hydration. Does NOT load the embedding/reranker
    # models — those load lazily on first use, same as the CLI.
    get_db()


# ----------------------------------------------------------------------
# System
# ----------------------------------------------------------------------

@app.get("/api/system/health")
def health():
    db = get_db()
    fp_current = _hash_documents(DOCUMENTS_DIR)
    fp_indexed = _read_manifest_fingerprint()
    return {
        "status": "ok",
        "index": {
            "status": "ready" if fp_indexed == fp_current else "stale",
            "document_count": len(set(db._doc_sources)),
            "chunk_count": len(db._doc_texts),
        },
        "models": {
            "embedding_model": embedder.MODEL_NAME,
            "embedding_loaded": embedder._get_model.cache_info().currsize > 0,
            "reranker_model": reranker.MODEL_NAME,
            "reranker_loaded": reranker._get_model.cache_info().currsize > 0,
        },
        "llm_configured": bool(rag_module.AI_API_KEY),
    }


@app.get("/api/system/config")
def config():
    return {
        "retrieval_modes": MODE_INFO,
        "pipeline_stages": PIPELINE_STAGES,
        "defaults": {
            "top_k": rag_module.TOP_K,
            "candidates": rag_module.CANDIDATES,
            "mmr_lambda": 0.7,
            "min_score": rag_module.MIN_SCORE,
        },
        "chunking": {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "models": {
            "embedding": embedder.MODEL_NAME,
            "reranker": reranker.MODEL_NAME,
            "llm": rag_module.AI_MODEL if rag_module.AI_API_KEY else None,
        },
        "bm25": {"k1": vectordb.BM25_K1, "b": vectordb.BM25_B},
        "rrf_k": vectordb.RRF_K,
        "prompt_version": rag_module.PROMPT_VERSION,
    }


# ----------------------------------------------------------------------
# Week 6 — evals
# ----------------------------------------------------------------------

@app.get("/api/eval/week6", response_model=Week6EvalResponse)
def eval_week6():
    return build_week6_summary()


@app.get("/api/eval/week6/label-queue", response_model=list[LabelQueueItem])
def eval_week6_label_queue():
    """Every judge-eligible case + its full context, for the blind human-labeling
    UI (/evals/label). Deliberately carries no judge verdict."""
    return get_label_queue()


@app.post("/api/eval/week6/labels", response_model=LabelSubmitResponse)
def eval_week6_submit_label(req: LabelSubmitRequest):
    """Save one real hand label to labels_25.json. This is what actually
    satisfies Week 6's blind-before-judge requirement — labels_25_draft_ai.json
    (Claude's own first-pass labels) is not read or used here."""
    return submit_label(req.id, req.verdict, req.reason, req.labeler)


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------

@app.get("/api/documents", response_model=DocumentsResponse)
def list_documents():
    db = get_db()
    counts = Counter(db._doc_sources)
    in_sync = _index_in_sync()

    docs = []
    for p in sorted(Path(DOCUMENTS_DIR).glob("*.pdf")):
        stat = p.stat()
        chunk_count = counts.get(p.name, 0)
        if chunk_count > 0 and in_sync:
            status = "indexed"
        elif not in_sync:
            status = "pending_reindex"
        else:
            status = "not_indexed"
        docs.append(DocumentInfo(
            filename=p.name,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            chunk_count=chunk_count,
            status=status,
        ))

    return DocumentsResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=len(db._doc_texts),
        index_in_sync=in_sync,
    )


@app.get("/api/documents/{filename}/file")
def get_document_file(filename: str):
    safe_name = Path(filename).name
    path = Path(DOCUMENTS_DIR) / safe_name
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(404, "Document not found")
    return FileResponse(path, media_type="application/pdf", filename=safe_name)


@app.delete("/api/documents/{filename}", response_model=DeleteResponse)
def delete_document(filename: str):
    safe_name = Path(filename).name
    path = Path(DOCUMENTS_DIR) / safe_name
    if not path.exists():
        raise HTTPException(404, "Document not found")

    path.unlink()
    db = get_db()
    stages = _rebuild_index(db)
    return DeleteResponse(
        deleted=safe_name,
        rebuild=RebuildResponse(
            stages=stages,
            document_count=len(set(db._doc_sources)),
            chunk_count=len(db._doc_texts),
        ),
    )


@app.post("/api/index/rebuild", response_model=RebuildResponse)
def rebuild_index():
    db = get_db()
    stages = _rebuild_index(db)
    return RebuildResponse(
        stages=stages,
        document_count=len(set(db._doc_sources)),
        chunk_count=len(db._doc_texts),
    )


@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF files are supported: {f.filename}")
        content = await f.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, f"{f.filename} exceeds the 50MB upload limit")
        dest = Path(DOCUMENTS_DIR) / Path(f.filename).name
        dest.write_bytes(content)
        saved.append(dest.name)

    db = get_db()

    def event_stream():
        yield sse({"stage": "upload", "status": "done", "detail": {"files": saved}})

        t0 = time.perf_counter()
        documents = load_pdfs(DOCUMENTS_DIR)
        yield sse({"stage": "parse", "status": "done", "duration_ms": ms(t0),
                   "detail": {"documents": len(documents)}})

        t1 = time.perf_counter()
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        yield sse({"stage": "chunk", "status": "done", "duration_ms": ms(t1),
                   "detail": {"chunks": len(chunks)}})

        t2 = time.perf_counter()
        vectors = embedder.embed_texts([c["text"] for c in chunks])
        yield sse({"stage": "embed", "status": "done", "duration_ms": ms(t2),
                   "detail": {"vectors": len(vectors)}})

        t3 = time.perf_counter()
        fingerprint = _hash_documents(DOCUMENTS_DIR)
        db.rebuild(chunks, vectors, fingerprint)
        yield sse({"stage": "index", "status": "done", "duration_ms": ms(t3),
                   "detail": {"chunk_count": len(chunks)}})

        yield sse({
            "stage": "ready", "status": "done",
            "detail": {
                "document_count": len(set(c["source"] for c in chunks)),
                "chunk_count": len(chunks),
            },
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----------------------------------------------------------------------
# Retrieval playground (retrieval only — no generation)
# ----------------------------------------------------------------------

@app.post("/api/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    rag = get_rag()
    t0 = time.perf_counter()
    hits, meta = rag.retrieve(
        req.question, top_k=req.top_k, mode=req.mode,
        candidates=req.candidates, mmr_lambda=req.mmr_lambda,
    )
    duration_ms = ms(t0)
    return RetrieveResponse(
        question=req.question, mode=req.mode, hits=hits,
        meta=RetrievalMeta(**meta), duration_ms=duration_ms,
    )


# ----------------------------------------------------------------------
# RAG query (retrieval + generation)
# ----------------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    rag = get_rag()
    result = rag.answer(req.question, top_k=req.top_k, mode=req.mode, mmr_lambda=req.mmr_lambda)
    write_trace(make_trace(req.question, result, mode=req.mode))
    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        grounded=result["grounded"],
        evidence=result["evidence"],
        meta=RetrievalMeta(**result["meta"]),
        best_score=result["best_score"],
        min_score=result["min_score"],
        timing_ms=result["timing_ms"],
        generation_method=result["generation_method"],
        prompt=result["prompt"],
    )


@app.post("/api/query/stream")
def query_stream(req: QueryRequest):
    """
    Same pipeline as /api/query, split into two real events (retrieval,
    then generation) so the frontend can render each stage as it actually
    completes, instead of faking a progress bar around one blocking call.
    """
    rag = get_rag()

    def gen():
        t0 = time.perf_counter()
        hits, meta = rag.retrieve(req.question, top_k=req.top_k, mode=req.mode, mmr_lambda=req.mmr_lambda)
        retrieval_ms = ms(t0)

        pool = meta.get("candidate_pool") or hits
        best_score = max((h.get("cosine") for h in pool if h.get("cosine") is not None), default=0.0)

        # Normalize through the same Pydantic models the non-streaming
        # endpoints use, so every hit always carries every field (null when
        # absent) instead of omitting keys the raw pipeline dicts don't set —
        # keeping the streamed contract identical to /api/retrieve and /api/query.
        yield sse({
            "event": "retrieved",
            "hits": [ChunkHit(**h).model_dump() for h in hits],
            "meta": RetrievalMeta(**meta).model_dump(),
            "best_score": best_score,
            "min_score": rag_module.MIN_SCORE,
            "duration_ms": retrieval_ms,
        })

        if best_score < rag_module.MIN_SCORE:
            write_trace(make_trace(
                req.question,
                {"answer": rag_module.DONT_KNOW_MESSAGE, "evidence": hits, "meta": meta, "grounded": False},
                mode=req.mode,
            ))
            yield sse({
                "event": "answer",
                "answer": rag_module.DONT_KNOW_MESSAGE,
                "grounded": False,
                "generation_method": "refused",
                "prompt": None,
                "duration_ms": 0,
            })
            return

        prompt = RAG._build_prompt(req.question, hits)
        t1 = time.perf_counter()
        generated = RAG._generate_with_llm(prompt)
        method = "llm"
        if generated is None:
            generated = RAG._synthesize_answer(req.question, hits)
            method = "fallback_synthesis"
        generation_ms = ms(t1)

        write_trace(make_trace(
            req.question,
            {"answer": generated, "evidence": hits, "meta": meta, "grounded": True},
            mode=req.mode,
        ))
        yield sse({
            "event": "answer",
            "answer": generated,
            "grounded": True,
            "generation_method": method,
            "prompt": prompt,
            "duration_ms": generation_ms,
        })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
