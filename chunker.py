
""" 
STEP 2:CHUNKING 
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split a long string into a list of overlapping chunks.

    Args:
        text:       the raw document text to split.
        chunk_size: how many characters go into each chunk.
        overlap:    how many characters of the previous chunk carry over
                    into the next one (context continuity).

    Returns:
        A list of chunk strings.
    """
    # Guard against silly inputs that would make the loop below infinite.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    # The stride is how far we slide the window forward each step.
    # If overlap=100 and chunk_size=500, each new chunk starts 400 chars in.
    stride = chunk_size - overlap

    # Slide a fixed-size window across the text until we reach the end.
    for start in range(0, len(text), stride):
        chunk = text[start : start + chunk_size]

        # Skip empties (e.g. whitespace-only edges) to keep the vector
        # store free of junk vectors.
        if chunk.strip():
            chunks.append(chunk)

        # Stop early once the window has covered the whole text so we
        # don't emit a repeated trailing chunk.
        if start + chunk_size >= len(text):
            break

    return chunks


def chunk_documents(documents: list[dict], **kwargs) -> list[dict]:
    chunks = []

    for doc in documents:
        for i, piece in enumerate(chunk_text(doc["text"], **kwargs)):
            chunks.append(
                {"text": piece, "source": doc["source"], "chunk_id": i}
            )

    return chunks


if __name__ == "__main__":
    # Quick manual test: python chunker.py
    sample = "The quick brown fox jumps over the lazy dog. " * 20
    pieces = chunk_text(sample, chunk_size=60, overlap=20)
    print(f"Created {len(pieces)} chunks from {len(sample)} chars")
