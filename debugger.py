

import sys

from embedder import embed_text
from evaluate import GOLDEN_SET, classify_failure, run_metrics_report
from rag import RAG
from vectordb import build_index


def visualize_retrieval(db, question: str, mode: str = "rerank", top_k: int = 3) -> dict:
    """
    Run every retrieval stage for one question and collect the results:
    semantic hits, BM25 hits, hybrid hits, reranked hits, final answer.
    """
    q_vec = embed_text(question)
    rag = RAG(db, mode=mode)

    semantic_hits = db.search(q_vec, top_k=top_k)
    keyword_hits = db.search_keyword(question, top_k=top_k * 2)
    hybrid_hits = db.hybrid_search(q_vec, question, top_k=top_k)

    result = rag.answer(question, mode=mode)

    return {
        "question": question,
        "semantic_hits": semantic_hits,
        "keyword_hits": keyword_hits,
        "hybrid_hits": hybrid_hits,
        "answer": result["answer"],
        "grounded": result["grounded"],
        "evidence": result["evidence"],
        "meta": result["meta"],
    }


def _fmt_hit(hit: dict, full: bool) -> str:
    text = hit["text"] if full else hit["text"][:90].replace("\n", " ") + "..."
    parts = [f"[{hit['source']}]"]
    for key, label in (
        ("cosine", "cosine"),
        ("score", "score"),
        ("retrieval_score", "retrieval"),
    ):
        if hit.get(key) is not None:
            parts.append(f"{label}={hit[key]:.3f}")
    return " ".join(parts) + f"\n    {text}"


def print_inspection(data: dict, full: bool = False) -> None:
    """Human-readable side-by-side inspection view."""
    w = 60
    print(f"\n{'=' * w}")
    print(f"QUESTION : {data['question']}")
    print(f"MODE     : {data['meta'].get('mode', '?')}")
    if data["meta"].get("rewritten"):
        print(f"REWRITTEN: {data['meta']['rewritten']}")
    if data["meta"].get("hypothetical"):
        print(f"HyDE     : {data['meta']['hypothetical'][:90]}...")
    print(f"{'=' * w}")

    print(f"\n--- WHAT WAS FETCHED: semantic (top-{len(data['semantic_hits'])}) ---")
    for i, h in enumerate(data["semantic_hits"], 1):
        print(f"  {i}. {_fmt_hit(h, full)}")

    print(f"\n--- WHAT WAS FETCHED: BM25 keyword (top-{len(data['keyword_hits'])}) ---")
    for i, h in enumerate(data["keyword_hits"][:5], 1):
        print(f"  {i}. {_fmt_hit(h, full)}")

    print(f"\n--- WHAT WAS FETCHED: hybrid RRF (top-{len(data['hybrid_hits'])}) ---")
    for i, h in enumerate(data["hybrid_hits"], 1):
        print(f"  {i}. {_fmt_hit(h, full)}")

    print(f"\n--- WHAT REACHED THE LLM (evidence, top-{len(data['evidence'])}) ---")
    for i, h in enumerate(data["evidence"], 1):
        print(f"  {i}. {_fmt_hit(h, full)}")

    print(f"\n--- FINAL ANSWER ---")
    print(f"Grounded: {data['grounded']}")
    print(data["answer"])


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    command = args[0]
    db = build_index()

    if command == "inspect":
        full = "--full" in args
        question = " ".join(a for a in args[1:] if not a.startswith("--"))
        if not question:
            question = "What brake fluid does the bike use?"
        mode = next((a for a in args[1:] if a in ("semantic", "hybrid", "rerank",
                                                  "hybrid_rerank", "rewrite", "hyde")), "rerank")
        print_inspection(visualize_retrieval(db, question, mode=mode), full=full)

    elif command == "classify":
        # e.g. python debugger.py classify "main fuse rating" "30 a"
        question = args[1]
        keywords = args[2:]
        if not keywords:
            print("Provide expected keywords: classify <question> kw1 kw2 ...")
            return
        data = visualize_retrieval(db, question)
        print_inspection(data)
        print(f"\nFailure type: {classify_failure(data['evidence'], data['answer'], data['grounded'], keywords) or 'OK (no failure)'}")

    elif command == "metrics":
        from evaluate import GOLDEN_SET, run_metrics_report

        run_metrics_report(db, GOLDEN_SET, modes=["semantic", "rerank"])

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
