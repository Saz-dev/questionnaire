"""
STEP 3: EMBEDDING
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

# Which model to use. all-MiniLM-L6-v2 is a well-known tiny model that
# produces 384-dim embeddings and downloads automatically on first use.
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer model exactly ONCE.

    Model loading is slow (~seconds), so we cache the result with
    lru_cache. Every call after the first reuses the same loaded model.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    
    if not texts:
        return []

    # encode() handles batching for us, so embedding 1 or 1000 texts is
    # just as easy. convert_to_numpy=False keeps it as a plain list.
    vectors = _get_model().encode(texts, convert_to_numpy=False)

    # numpy arrays aren't JSON-serializable, so normalise to plain lists.
    return [v.tolist() for v in vectors]


def embed_text(text: str) -> list[float]:
    """Convenience wrapper for embedding a single string."""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Quick sanity check: similar sentences should land close together.
    a = embed_text("Claim forms must be signed")
    b = embed_text("You need to sign the claim form")
    c = embed_text("The sky is blue today")

    import math

    def cosine(x, y):
        dot = sum(i * j for i, j in zip(x, y))
        nx = math.sqrt(sum(i * i for i in x))
        ny = math.sqrt(sum(j * j for j in y))
        return dot / (nx * ny)

    print(f"a vs b (similar):  {cosine(a, b):.3f}")
    print(f"a vs c (different): {cosine(a, c):.3f}")
