
import pathlib
import re
import sys

from chunker import chunk_documents
from embedder import embed_texts
from loader import load_pdfs
from rag import RAG
from vectordb import VectorDB

TOP_K = 3
REPORT_PATH = pathlib.Path("WEEK4_REPORT.md")

# ----------------------------------------------------------------------
# The golden test set. "keywords" = facts that MUST be in a chunk for it
# to count as "the right document". Substring matching on lowercased text.
# ----------------------------------------------------------------------

GOLDEN_SET = [
    {
        "question": "What is the recommended engine oil for the motorcycle?",
        "keywords": ["10w-30", "5w-30", "sj"],
        "note": "Oil spec (maintenance + Specifications sections)",
    },
    {
        "question": "What is the minimum tread depth for the front tyre?",
        "keywords": ["1.5 mm"],
        "note": "Specifications - Service Data",
    },
    {
        "question": "What is the main fuse rating?",
        "keywords": ["30 a"],
        "note": "Specifications - Fuses",
    },
    {
        "question": "How many links does the standard drive chain have?",
        "keywords": ["104"],
        "note": "Specifications - Service Data",
    },
    {
        "question": "What is the recommended spark plug for this bike?",
        "keywords": ["mr6k-9"],
        "note": "Specifications - Service Data (exact part number)",
    },
    {
        "question": "Which brake fluid should I use?",
        "keywords": ["dot 4"],
        "note": "Brake Fluid section / Specifications",
    },
    {
        "question": "What is the torque for the rear axle nut?",
        "keywords": ["88 n"],
        "note": "Drive chain adjustment / Torque table",
    },
    {
        "question": "What fuel octane should I use?",
        "keywords": ["91 or higher"],
        "note": "Refuelling section (fact appears as '(RON) 91 or higher')",
    },
    {
        "question": "What is the idle speed?",
        "keywords": ["1,000 ± 100 rpm"],
        "note": "Specifications - Service Data",
    },
    {
        "question": "What is the tyre pressure for the front when riding alone?",
        "keywords": ["200 kpa"],
        "note": "Specifications - Service Data",
    },
    {
        "question": "What is the correct drive chain slack?",
        "keywords": ["25 - 35 mm"],
        "note": "Drive chain section / Specifications",
    },
    {
        "question": "What is the clutch lever freeplay?",
        "keywords": ["10 - 20 mm"],
        "note": "Clutch section / Specifications",
    },
    {
        "question": "What is the first step to change the spark plug?",
        "keywords": ["spark plug cap"],
        "answer_keywords": ["disconnect"],
        "note": "Spark plug procedure (procedural question)",
    },
    {
        "question": "How long is the warranty period in India?",
        "keywords": ["36 months"],
        "note": "Warranty Policy",
    },
    {
        "question": "What battery does the CB350 use?",
        "keywords": ["ytz7"],
        "note": "Specifications - Battery (exact model code)",
    },
    {
        "question": "What type is the headlight?",
        "keywords": ["headlight led"],
        "answer_keywords": ["led type"],
        "note": "Specifications - Bulbs (fact appears as 'Headlight LED')",
    },
]

# Questions that are NOT in the manual: the app must refuse them.
REFUSAL_SET = [
    {"question": "What is the top speed of the CB350?", "note": "not in manual"},
    {"question": "What is the capital of France?", "note": "not in manual"},
]


def build_index() -> VectorDB:
    docs = load_pdfs()
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    db = VectorDB()
    db.add_chunks(chunks, embed_texts([c["text"] for c in chunks]))
    return db


# ----------------------------------------------------------------------
# Hit detection + metrics
# ----------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Canonical form for fact matching: lowercase, collapse any run of
    whitespace to a single space, unify dashes. PDF extraction scatters
    facts across line breaks ("(RON)\\n91 or higher") — this makes a
    substring match see the fact as it reads on the page.
    """
    return re.sub(r"\s+", " ", text.lower()).replace("‑", "-").replace("–", "-")


def is_right_doc(hit: dict, keywords: list[str]) -> bool:
    """A chunk is 'the right document' if it contains any expected fact."""
    text = normalize(hit["text"])
    return any(normalize(kw) in text for kw in keywords)


def rank_metrics(hits: list[dict], keywords: list[str]) -> dict:
    """
    Per-question retrieval metrics over the ranked top-k hits:
      hit@1: is the FIRST result the right document?
      hit@3: is the right document anywhere in the top-3?
      mrr:   reciprocal rank of the first right document (1, 0.5, 0.33...)
      recall@3: fraction of expected facts found in the top-3.
    """
    rank = None
    found: set[str] = set()
    for i, h in enumerate(hits, 1):
        text = normalize(h["text"])
        for kw in keywords:
            if normalize(kw) in text:
                found.add(normalize(kw))
                if rank is None:
                    rank = i
    return {
        "hit@1": 1.0 if rank == 1 else 0.0,
        "hit@3": 1.0 if rank is not None else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "recall@3": len(found) / len(keywords),
        "first_right_rank": rank,
    }


def aggregate(per_case: list[dict]) -> dict:
    """Average per-question metrics into one table row."""
    n = len(per_case)
    keys = ("hit@1", "hit@3", "mrr", "recall@3")
    return {k: sum(c["metrics"][k] for c in per_case) / n for k in keys}


def retrieval_metrics(db: VectorDB, cases: list[dict], mode: str) -> list[dict]:
    """
    Retrieval-only metrics (no LLM): for every question, run the given
    mode and score the ranked hits against the expected keywords.
    """
    rag = RAG(db, mode=mode)
    per_case = []
    for tc in cases:
        hits, _meta = rag.retrieve(tc["question"], top_k=TOP_K)
        per_case.append(
            {
                "question": tc["question"],
                "keywords": tc["keywords"],
                "hits": hits,
                "metrics": rank_metrics(hits, tc["keywords"]),
            }
        )
    return per_case


# ----------------------------------------------------------------------
# Failure classification
# ----------------------------------------------------------------------

def classify_failure(evidence: list[dict], answer: str, grounded: bool,
                     keywords: list[str], answer_keywords: list[str] | None = None) -> str | None:
    """
    The two kinds of wrong, with evidence:

    - "wrong_document": the right chunk is NOT in the evidence (or nothing
      was retrieved). Retrieval bug -> fix retrieval.
    - "right_document_wrong_answer": the right chunk IS in the evidence but
      the answer still misses the expected fact. Generation bug -> fix the
      prompt/LLM.
    - None: answer contains the expected fact.

    `answer_keywords` (optional) are the facts the ANSWER must contain;
    they can differ from the chunk facts (e.g. the answer says "LED type"
    while the chunk says "Headlight LED").
    """
    right_doc_fetched = any(is_right_doc(h, keywords) for h in evidence)
    expected_in_answer = answer_keywords or keywords
    answer_correct = any(normalize(kw) in normalize(answer) for kw in expected_in_answer)

    if not grounded:
        return "wrong_document"
    if not right_doc_fetched:
        return "wrong_document"
    if not answer_correct:
        return "right_document_wrong_answer"
    return None


# ----------------------------------------------------------------------
# The week's experiment: baseline vs ONE change (reranking)
# ----------------------------------------------------------------------

def run_experiment(db: VectorDB) -> None:
    """The deliverable: failing set -> classify -> one change -> measure."""
    modes = ["semantic", "rerank"]  # baseline = last week's app; one change = rerank
    label = {"semantic": "BASELINE (semantic only)", "rerank": "ONE CHANGE (rerank)"}

    print("=" * 70)
    print("WEEK 4 EXPERIMENT — one change, measured, with numbers")
    print("=" * 70)

    # 1) Retrieval metrics before/after (no LLM needed).
    rows = {}
    for mode in modes:
        rows[mode] = retrieval_metrics(db, GOLDEN_SET, mode)
        agg = aggregate(rows[mode])
        print(f"\n{label[mode]}: hit@1={agg['hit@1']:.2f} hit@3={agg['hit@3']:.2f} "
              f"MRR={agg['mrr']:.2f} recall@3={agg['recall@3']:.2f}")

    # 2) Failure classification before/after (needs the real answer).
    classified = {mode: [] for mode in modes}
    rag = {mode: RAG(db, mode=mode) for mode in modes}
    for tc in GOLDEN_SET:
        for mode in modes:
            result = rag[mode].answer(tc["question"], mode=mode)
            failure = classify_failure(
                result["evidence"], result["answer"], result["grounded"],
                tc["keywords"], tc.get("answer_keywords"),
            )
            entry = {
                **tc,
                "failure": failure,
                "answer": result["answer"],
                "evidence": result["evidence"],
                "grounded": result["grounded"],
            }
            # For reranking modes: was the right chunk even in the candidate
            # pool? (pool-miss = the reranker never had a chance).
            pool = result["meta"].get("candidate_pool", [])
            entry["right_doc_in_pool"] = (
                any(is_right_doc(h, tc["keywords"]) for h in pool) if pool else None
            )
            classified[mode].append(entry)

    _write_report(db, rows, classified)

    # 3) Summary for the console.
    before = aggregate(rows["semantic"])
    after = aggregate(rows["rerank"])
    print("\n" + "=" * 70)
    print("BEFORE/AFTER — hit-rate@3 (right doc shows up in top-3)")
    print(f"  semantic only : {before['hit@3']:.2%}")
    print(f"  + rerank      : {after['hit@3']:.2%}")
    print(f"  delta         : {after['hit@3'] - before['hit@3']:+.2%}")
    print("=" * 70)
    print(f"Full report written to {REPORT_PATH}")


def _preview(text: str, width: int = 70) -> str:
    return text.replace("\n", " ")[:width] + ("..." if len(text) > width else "")


def _write_report(db: VectorDB, rows: dict, classified: dict) -> None:
    modes = ["semantic", "rerank"]
    label = {"semantic": "baseline", "rerank": "rerank"}
    before, after = aggregate(rows["semantic"]), aggregate(rows["rerank"])

    lines = []
    a = lines.append
    a("# Week 4 Report — Retrieval & RAG Debugging")
    a("")
    a("Student task: take failing questions, separate the two kinds of wrong, "
      "make **one** change, and prove it with a number.")
    a("")
    a("## 1. The two kinds of wrong")
    a("")
    a("- **wrong_document** — the right chunk never made it into the evidence. "
      "Retrieval bug; a smarter LLM would change nothing.")
    a("- **right_document_wrong_answer** — the right chunk *was* fetched, but "
      "the answer still missed the fact. Generation bug.")
    a("")
    a("## 2. The failing set (16 questions, facts the right chunk must contain)")
    a("")
    a("| Question | Expected fact | Baseline outcome | Failure type |")
    a("|---|---|---|---|")
    for c in classified["semantic"]:
        q = c["question"].replace("|", "\\|")
        kw = ", ".join(c["keywords"]).replace("|", "\\|")
        outcome = "refused" if not c["grounded"] else "answered"
        failure = c["failure"] or "OK"
        a(f"| {q} | `{kw}` | {outcome} | {failure} |")
    a("")
    a("## 3. The ONE change I made")
    a("")
    a("Added a **cross-encoder reranker** on top of the existing semantic search "
      "(`cross-encoder/ms-marco-MiniLM-L-6-v2`; drop-in for Cohere Rerank / "
      "BGE-Reranker). Pipeline: semantic search top-50 → cross-encoder scores "
      "query×chunk pairs together → keep top-3. Nothing else changed.")
    a("")
    a("## 4. Before / after — measured, not eyeballed")
    a("")
    a("| Metric | Before (semantic only) | After (one change: rerank) | Delta |")
    a("|---|---|---|---|")
    for k, name in (("hit@1", "hit-rate@1"), ("hit@3", "hit-rate@3"),
                    ("mrr", "MRR"), ("recall@3", "recall@3")):
        d = after[k] - before[k]
        a(f"| {name} | {before[k]:.2%} | {after[k]:.2%} | {d:+.2%} |")
    a("")
    a("## 5. Failures the change did NOT fix")
    a("")
    a("| Question | Failure after rerank | Right chunk in the 50-candidate pool? | What was fetched instead (top-3) |")
    a("|---|---|---|---|")
    unfixed = [c for c in classified["rerank"] if c["failure"] == "wrong_document"]
    still_bad_answer = [c for c in classified["rerank"]
                        if c["failure"] == "right_document_wrong_answer"]
    for c in unfixed:
        fetched = "; ".join(f"`{_preview(h['text'], 40)}`" for h in c["evidence"][:3])
        in_pool = "yes (reranker missed it)" if c.get("right_doc_in_pool") else "no (never retrieved)"
        a(f"| {c['question'].replace('|', '\\|')} | wrong_document | {in_pool} | {fetched} |")
    if not unfixed:
        a("| (none) | — | — | — |")
    a("")
    pool_misses = sum(1 for c in unfixed if not c.get("right_doc_in_pool"))
    pool_hits = len(unfixed) - pool_misses
    a("**Why they are not fixed:** reranking can only re-order what retrieval "
      "already found. "
      + (f"{pool_misses} of the {len(unfixed)} unfixed questions had their right "
         "chunk OUTSIDE the 50-candidate pool — no reranker can fix a "
         "retrieval-pool miss; those need hybrid search or query rewriting instead. "
         if pool_misses else "")
      + (f"{pool_hits} of the {len(unfixed)} unfixed questions DID have their "
         "right chunk inside the pool, but the cross-encoder still ranked it "
         "below top-3 — a stronger reranker or a larger top_k is the next "
         "lever there. "
         if pool_hits else ""))
    if still_bad_answer:
        a("Questions the reranker DID fetch correctly but the answer still missed "
          "(generation-side):")
        a("")
        for c in still_bad_answer:
            a(f"- **{c['question']}** — expected `{'`/`'.join(c['keywords'])}`; "
              f"answer: {_preview(c['answer'], 90)}")
    a("")
    a("## 6. Method & limitations")
    a("")
    a("- \"Right document\" = any retrieved chunk containing the expected fact "
      "(substring match on normalized text — lowercase, whitespace collapsed). "
      "Simple and auditable; may count a near-miss chunk as a hit.")
    a("- Chunk-boundary caveat: fixed-size chunks can split a fact across two "
      "chunks (e.g. 'Di|sconnect the spark plug cap'), so that chunk can never "
      "match the keyword. Where that happened, the keyword was chosen from the "
      "part of the fact that survives inside one chunk.")
    a("- Some questions have separate answer-side keywords (what the LLM "
      "answer must contain) because the answer phrases the fact differently "
      "than the chunk does (e.g. chunk 'Headlight LED' vs answer 'LED type').")
    a("- Rerank candidates: top-50 semantic. top_k=3, MIN_SCORE=0.35.")
    a("- Generation: Groq `openai/gpt-oss-120b`, temperature 0.2.")
    a("- All numbers in section 4 are computed on the same 16 questions, same "
      "index — only the retrieval strategy differs.")
    a("")
    a("## 7. Bonus experiments (NOT the one change — reported for context)")
    a("")
    a("The toolkit also ships hybrid search (BM25+RRF), query rewriting, HyDE "
      "and MMR. Retrieval metrics for interest:")
    a("")
    a("| Mode | hit@1 | hit@3 | MRR | recall@3 |")
    a("|---|---|---|---|---|")
    for mode in ("hybrid", "hybrid_rerank", "rewrite", "hyde", "mmr"):
        m = retrieval_metrics(db, GOLDEN_SET, mode)
        agg = aggregate(m)
        a(f"| {mode} | {agg['hit@1']:.2%} | {agg['hit@3']:.2%} | "
          f"{agg['mrr']:.2%} | {agg['recall@3']:.2%} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_PATH} ({len(lines)} lines)")


def run_metrics_report(db: VectorDB, cases: list[dict], modes: list[str]) -> None:
    print(f"{'mode':<14} {'hit@1':>7} {'hit@3':>7} {'MRR':>7} {'recall@3':>9}")
    for mode in modes:
        m = aggregate(retrieval_metrics(db, cases, mode))
        print(f"{mode:<14} {m['hit@1']:>6.2%} {m['hit@3']:>6.2%} "
              f"{m['mrr']:>6.2%} {m['recall@3']:>8.2%}")


def main() -> None:
    args = sys.argv[1:]
    db = build_index()

    if args and args[0] == "metrics":
        modes = args[1:] or ["semantic", "hybrid", "rerank", "hybrid_rerank",
                             "rewrite", "hyde", "mmr"]
        run_metrics_report(db, GOLDEN_SET, modes)
    else:
        run_experiment(db)


if __name__ == "__main__":
    main()
