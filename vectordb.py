"""
STEP 4: VECTOR DATABASE (STORAGE + RETRIEVAL)
=============================================
We now have thousands of chunks, each with an embedding vector. To answer
"what's closest to this question?" we need a vector database.

A vector DB stores (vector, payload) pairs and answers similarity queries.
A brute-force scan works for small demos, but real systems index vectors
(e.g. HNSW) so search is fast even with millions of entries.

We use Qdrant:
  - `QdrantClient(":memory:")` runs the whole DB in-process with zero
    setup — perfect for learning. No server, no Docker, no credentials.
  - `upsert` adds/overwrites points.
  - `query_points` returns the nearest points to a query vector.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Each chunk becomes one "point" with:
#   - a vector embedding (the numbers)
#   - a payload (the readable text + metadata we want back on retrieval)
COLLECTION_NAME = "bike_maintenance_docs"

# Where vectors live and how "nearness" is measured. Cosine similarity
# compares directions (semantic meaning) rather than magnitude (length),
# which is the standard choice for text embeddings.
VECTOR_SIZE = 384  # matches all-MiniLM-L6-v2's output dimension


class VectorDB:
    """A tiny wrapper around an in-memory Qdrant collection."""

    def __init__(self):
        # ":memory:" = self-contained in-memory store, wiped when the
        # script exits. To persist across runs you'd swap in a server URL.
        self.client = QdrantClient(":memory:")
        # Create the collection and declare the vector configuration.
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    def add_chunks(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        """
        Store every chunk (with its embedding) in the collection.

        Args:
            chunks:  the list from chunker.chunk_documents()
            vectors: parallel list of embeddings, one per chunk

        Returns:
            Number of points inserted.
        """
        points = [
            PointStruct(
                # Qdrant needs a unique integer id per point. Enumerate is
                # fine here since we're building from a fresh collection.
                id=i,
                vector=vectors[i],
                # The payload rides along and comes back with results, so
                # we can show the user WHICH text matched and WHERE it's from.
                payload={
                    "text": chunks[i]["text"],
                    "source": chunks[i]["source"],
                    "chunk_id": chunks[i]["chunk_id"],
                },
            )
            for i in range(len(chunks))
        ]

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    def search(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        """
        Find the top_k chunks nearest to a query vector.

        Args:
            query_vector: embedding of the user's question.
            top_k:        how many matches to return.

        Returns:
            List of results, each dict containing the payload text/source
            plus the `score` (cosine similarity, higher = more related).
        """
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,  # bring back the readable text/metadata
        ).points

        return [
            {
                "text": r.payload["text"],
                "source": r.payload["source"],
                "score": r.score,
            }
            for r in results
        ]


if __name__ == "__main__":
    # Quick manual test: python vectordb.py
    from embedder import embed_text

    db = VectorDB()
    db.add_chunks(
        [{"text": "Claim forms must be signed.", "source": "test", "chunk_id": 0}],
        [embed_text("Claim forms must be signed.")],
    )
    hits = db.search(embed_text("What must I sign?"))
    print("Top hit:", hits[0]["text"], "| score:", round(hits[0]["score"], 3))
