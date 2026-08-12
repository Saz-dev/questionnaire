"""
STEP 5: RAG — RETRIEVAL-AUGMENTED GENERATION
============================================
This is the actual "RAG" step that ties everything together:

  1. EMBED the user's question.
  2. RETRIEVE the most relevant chunks from the vector DB (semantic search).
  3. GENERATE an answer conditioned on those chunks (with an LLM).

The trick of RAG: instead of asking an LLM from memory (where it may
hallucinate, or know nothing about your private documents), we feed the
retrieved evidence INTO the prompt. The LLM's only job is to read and
summarise that evidence — so answers are grounded in your data.

Generation uses OpenAI if an OPENAI_API_KEY is set in the environment
(the `openai` package is already installed). Otherwise it falls back to
returning the retrieved context verbatim, so the whole pipeline still
runs locally with zero dependencies.
"""

import os

from dotenv import load_dotenv

from embedder import embed_text
from vectordb import VectorDB

# Load AI_API_KEY / AI_MODEL / APP_NAME from .env (no-op if not present).
load_dotenv()

# How many chunks to feed the LLM as "evidence".
TOP_K = 3

# LLM provider config (Groq's OpenAI-compatible API).
AI_API_KEY = os.environ.get("AI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "llama-3.3-70b-versatile")
AI_BASE_URL = "https://api.groq.com/openai/v1"
AI_APP_NAME = os.environ.get("APP_NAME", "Smart CLI")


class RAG:
    """End-to-end retrieval-augmented generation over one VectorDB."""

    def __init__(self, db: VectorDB):
        self.db = db

    def _retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Return the top_k chunks most similar to the question."""
        q_vector = embed_text(question)
        return self.db.search(q_vector, top_k=top_k)

    @staticmethod
    def _build_prompt(question: str, hits: list[dict]) -> str:
        """
        Compose the LLM prompt. The retrieved chunks are inserted between
        <context> tags; the instruction tells the model to rely ONLY on
        that context and to cite sources — this is what keeps RAG grounded
        and auditable.
        """
        evidence = "\n\n---\n\n".join(
            f"Source: {h['source']}\n{h['text']}" for h in hits
        )
        return f"""
You are a helpful assistant answering questions about Honda bike maintenance.

Use ONLY the context below to answer. If the context does not contain
the answer, say you don't know. Mention which source document each part
of your answer comes from.

<context>
{evidence}
</context>

Question: {question}
Answer:
""".strip()

    @staticmethod
    def _generate_with_llm(prompt: str) -> str | None:
        """Call the AI provider's chat completions API via Groq. None if no key."""
        if not AI_API_KEY:
            return None  # no key -> caller falls back to context-only answer

        # Groq exposes an OpenAI-compatible endpoint, so the `openai` client
        # works by just swapping the base_url. Imported lazily so local
        # mode needs nothing.
        from openai import OpenAI

        client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are {AI_APP_NAME}, an assistant answering "
                    "questions about honda bike maintenance. Use ONLY the context "
                    "provided to answer.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # low temperature -> factual, reproducible answers
        )
        return response.choices[0].message.content

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        """
        The full RAG flow: retrieve evidence, then generate an answer.

        Returns a dict with the answer plus the evidence used, so callers
        can show retrieval debugging info.
        """
        # 1) Find relevant evidence in the vector store.
        hits = self._retrieve(question, top_k=top_k)

        # 2) Turn evidence + question into a prompt.
        prompt = self._build_prompt(question, hits)

        # 3) Try to generate with an LLM; degrade gracefully to raw context.
        generated = self._generate_with_llm(prompt)
        if generated is None:
            generated = (
                "(No AI_API_KEY set — returning retrieved context instead. "
                f"Set AI_API_KEY in .env to get LLM-generated answers via {AI_MODEL}.)"
            )

        return {"answer": generated, "evidence": hits}


if __name__ == "__main__":
    # Quick manual test: python rag.py
    from loader import load_pdfs
    from chunker import chunk_documents
    from embedder import embed_texts

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    db = VectorDB()
    db.add_chunks(chunks, embed_texts([c["text"] for c in chunks]))

    result = RAG(db).answer("What documents are needed for a total loss claim?")
    print("ANSWER:", result["answer"][:400])
