from functools import lru_cache

from sentence_transformers import CrossEncoder

from embedder import embed_texts

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Production alternatives (drop-in via model name / API):
#   - Cohere Rerank:    API-based, `cohere.Rerank(model="rerank-multilingual-v3.0")`
#   - BGE-Reranker:     "BAAI/bge-reranker-base" (heavier, ~1.1 GB)
# ms-marco-MiniLM is ~90 MB and fast on CPU — right size for this course.


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Load the CrossEncoder exactly once (model load is slow)."""
    return CrossEncoder(MODEL_NAME)


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1] within this candidate pool. A raw
    cross-encoder logit and a raw cosine similarity are on completely
    different scales, so they can't be blended without this."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [0.5 for _ in values]
    return [(v - lo) / span for v in values]


def rerank(query: str, candidates: list[dict], top_k: int = 3, ce_weight: float = 0.5) -> list[dict]:
    """
    Re-order a candidate pool by cross-encoder relevance to the query,
    blended with the original retrieval cosine.

    Trusting the cross-encoder alone is not safe on this corpus: verified on
    two real queries ("what are the tools provided with the cb350 hness" and
    "what is the main fuse rating?"), ms-marco-MiniLM-L-6-v2 ranked a
    thematically-unrelated chunk (display/backlight settings) above the chunk
    that actually contained the answer, even though the correct chunk was
    the #2 candidate by raw cosine in both cases. The cross-encoder appears
    to do poorly on this manual's terse, list-formatted spec/procedure text.
    Blending keeps the cross-encoder as a real signal (`ce_weight`, default
    an even 50/50) without letting one bad judgment bury a chunk the cheap
    embedding search was confident about.

    Each candidate keeps `retrieval_score` (the pre-rerank cosine/RRF score)
    and gains `cross_encoder_score` (the raw, unblended cross-encoder logit)
    alongside the blended `score` used for the final ordering, so an
    inspection view can still show all three numbers separately.
    """
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    ce_scores = [float(s) for s in _get_model().predict(pairs)]
    cosines = [c.get("cosine") if c.get("cosine") is not None else c.get("score", 0.0) for c in candidates]

    norm_ce = _normalize(ce_scores)
    norm_cosine = _normalize(cosines)

    reranked = []
    for c, ce, nce, ncos in zip(candidates, ce_scores, norm_ce, norm_cosine):
        blended = ce_weight * nce + (1 - ce_weight) * ncos
        reranked.append(
            {
                **c,
                "retrieval_score": c.get("score"),
                "cross_encoder_score": ce,
                "score": blended,
            }
        )
    reranked.sort(key=lambda x: -x["score"])
    return reranked[:top_k]


def mmr_rerank(
    query_vector: list[float],
    candidates: list[dict],
    top_k: int = 3,
    lambda_: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance: relevance minus similarity to what was
    already picked.

        mmr = lambda * sim(query, d) - (1 - lambda) * max(sim(d, picked))

    lambda_ = 1.0 is pure relevance (plain semantic order); lambda_ = 0.0
    is pure diversity. 0.7 keeps relevance dominant while still breaking
    up near-duplicate chunks.
    """
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    vectors = embed_texts(texts)

    picked: list[dict] = []
    picked_vectors: list[list[float]] = []
    remaining = list(range(len(candidates)))

    def cosine(a: list[float], b: list[float]) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    while remaining and len(picked) < top_k:
        best_i, best_score = -1, float("-inf")
        for i in remaining:
            relevance = cosine(query_vector, vectors[i])
            if picked_vectors:
                redundancy = max(cosine(vectors[i], p) for p in picked_vectors)
            else:
                redundancy = 0.0
            mmr = lambda_ * relevance - (1 - lambda_) * redundancy
            if mmr > best_score:
                best_i, best_score = i, mmr

        picked.append({**candidates[best_i], "mmr_score": best_score})
        picked_vectors.append(vectors[best_i])
        remaining.remove(best_i)

    return picked


if __name__ == "__main__":
    from vectordb import build_index
    from embedder import embed_text

    db = build_index()
    pool = db.search(embed_text("brake fluid"), top_k=10)
    print("Semantic top-3 first:")
    for r in pool[:3]:
        print(f"  {r['score']:.3f} -> {r['text'][:60]}...")

    print("\nCross-encoder rerank top-3:")
    for r in rerank("What brake fluid does the bike use?", pool, top_k=3):
        print(f"  {r['score']:.3f} (retrieval {r['retrieval_score']:.3f}) -> {r['text'][:60]}...")
