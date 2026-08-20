import math
import re
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "bike_maintenance_docs"
VECTOR_SIZE = 384

# BM25 parameters (Okapi). k1 controls term-frequency saturation, b controls
# document-length normalisation.
BM25_K1 = 1.5
BM25_B = 0.75

# RRF constant: the "+60" gives every result a healthy base score so the
# first result of each list always contributes.
RRF_K = 60

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-:./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """
    Lowercase and split into searchable tokens.

    Keeps alphanumeric runs together with separators so codes like
    "MR6K-9", "DID520VF4", "5W-30" or "88 N" stay intact — those are
    exactly the exact-match terms keyword search exists for.
    """
    return _TOKEN_RE.findall(text.lower())


def build_index() -> "VectorDB":
    from loader import load_pdfs
    from chunker import chunk_documents
    from embedder import embed_texts

    documents = load_pdfs()
    chunks = chunk_documents(documents, chunk_size=500, overlap=100)
    db = VectorDB()
    db.add_chunks(chunks, embed_texts([c["text"] for c in chunks]))
    print(f"-> {len(chunks)} vectors stored")
    return db


class VectorDB:
    """A tiny wrapper around an in-memory Qdrant collection + BM25 index."""

    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        # BM25 stats, computed once when chunks are added.
        self._doc_texts: list[str] = []
        self._doc_sources: list[str] = []
        self._doc_chunk_ids: list[str] = []
        self._doc_freq: Counter[str] = Counter()
        self._doc_lengths: list[int] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        points = [
            PointStruct(
                id=i,
                vector=vectors[i],
                payload={
                    "text": chunks[i]["text"],
                    "source": chunks[i]["source"],
                    # Stable, citable ID ("source::index") so a golden set can
                    # pin down "the correct chunk" instead of just a keyword.
                    "chunk_id": f"{chunks[i]['source']}::{chunks[i].get('chunk_id', i)}",
                },
            )
            for i in range(len(chunks))
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

        # One pass over the corpus to collect BM25 statistics.
        self._doc_texts = [c["text"] for c in chunks]
        self._doc_sources = [c["source"] for c in chunks]
        self._doc_chunk_ids = [
            f"{chunks[i]['source']}::{chunks[i].get('chunk_id', i)}" for i in range(len(chunks))
        ]
        self._doc_lengths = [len(tokenize(t)) for t in self._doc_texts]
        self._doc_freq = Counter(
            term
            for doc_tokens in map(tokenize, self._doc_texts)
            for term in set(doc_tokens)
        )
        return len(points)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        """Semantic search: nearest chunks by cosine similarity."""
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "text": r.payload["text"],
                "source": r.payload["source"],
                "chunk_id": r.payload.get("chunk_id"),
                "score": r.score,
                "cosine": r.score,
            }
            for r in results
        ]

    def search_keyword(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25 keyword search.

        A chunk scores high when it contains the query terms often (tf),
        when those terms are rare in the corpus (idf), and when the chunk
        is not absurdly long (b normalises by average length).
        """
        query_terms = tokenize(query)
        if not query_terms or not self._doc_texts:
            return []

        n = len(self._doc_texts)
        avg_dl = sum(self._doc_lengths) / n

        # IDF: rarer terms carry more weight. The +0.5 smoothing keeps
        # terms that appear in every document from vanishing.
        idf = {
            term: math.log(1 + (n - self._doc_freq[term] + 0.5)
                           / (self._doc_freq[term] + 0.5))
            for term in query_terms
        }

        scored = []
        for i, text in enumerate(self._doc_texts):
            tokens = self._doc_lengths[i]
            if tokens == 0:
                continue

            tf = Counter(tokenize(text))
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                tf_component = (freq * (BM25_K1 + 1)) / (
                    freq + BM25_K1 * (1 - BM25_B + BM25_B * tokens / avg_dl)
                )
                score += idf[term] * tf_component

            if score > 0:
                scored.append(
                    {
                        "text": text,
                        "source": self._doc_sources[i],
                        "chunk_id": self._doc_chunk_ids[i],
                        "score": score,
                    }
                )

        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

    def hybrid_search(
        self, query_vector: list[float], query_text: str, top_k: int = 5
    ) -> list[dict]:
        """
        Hybrid search: semantic + BM25 fused via Reciprocal Rank Fusion.

        Instead of merging raw scores (semantic cosine and BM25 are on
        different scales), RRF merges *ranks*: each chunk earns
        1/(RRF_K + rank) from every list it appears in. A chunk both
        retrievers rank highly rises to the top.
        """
        pool = top_k * 3
        vec_results = self.search(query_vector, top_k=pool)
        kw_results = self.search_keyword(query_text, top_k=pool)

        rrf_scores: dict[int, float] = {}
        by_index: dict[int, dict] = {}
        for results in (vec_results, kw_results):
            for rank, r in enumerate(results):
                idx = self._index_of(r["text"])
                if idx is None:
                    continue
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
                by_index.setdefault(idx, r)

        fused = [
            {**by_index[idx], "score": rrf, "cosine": by_index[idx].get("cosine")}
            for idx, rrf in rrf_scores.items()
        ]
        fused.sort(key=lambda x: -x["score"])
        return fused[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _index_of(self, text: str) -> int | None:
        """Find a chunk index by exact text (used to align RRF lists)."""
        for i, doc in enumerate(self._doc_texts):
            if doc == text:
                return i
        return None

    def find_chunk_id(self, keyword: str) -> str | None:
        """
        Ground-truth lookup for building a golden set: which chunk_id in the
        WHOLE corpus contains this fact? (Not a search result — a scan, so
        the golden set's "known-correct chunk_id" doesn't depend on any
        retrieval strategy being right.)
        """
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.lower()).replace("‑", "-").replace("–", "-")

        needle = norm(keyword)
        for i, text in enumerate(self._doc_texts):
            if needle in norm(text):
                return self._doc_chunk_ids[i]
        return None

    def corpus_contains(self, keyword: str) -> bool:
        """Does the fact exist ANYWHERE in the indexed corpus at all?"""
        return self.find_chunk_id(keyword) is not None


if __name__ == "__main__":
    from embedder import embed_text

    db = build_index()
    result = db.search(embed_text("What must I sign?"))
    print(f"Top hit: {result[0]['text'][:60]}... | score: {round(result[0]['score'], 3)}")

    print("\nBM25 search for 'main fuse 30':")
    kw = db.search_keyword("main fuse 30 A", top_k=3)
    for r in kw:
        print(f"  [{r['score']:.3f}] {r['text'][:70]}...")

    print("\nHybrid search for 'main fuse 30 A':")
    hy = db.hybrid_search(embed_text("main fuse 30 A"), "main fuse 30 A", top_k=3)
    for r in hy:
        print(f"  [{r['score']:.3f}] {r['text'][:70]}...")
