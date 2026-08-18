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

Generation uses OpenAI if an AI_API_KEY is set in the environment
(the `openai` package is already installed via groq). Otherwise it falls
back to presenting the retrieved context directly, still running locally
with zero additional dependencies.
"""

import os

from dotenv import load_dotenv

from embedder import embed_text
from vectordb import VectorDB

# Load AI_API_KEY / AI_MODEL / APP_NAME from .env (no-op if not present).
load_dotenv()

# How many chunks to feed the LLM as "evidence".
TOP_K = 3

# Minimum cosine similarity a retrieved chunk needs before we trust it
# enough to generate an answer from it. Below this, the question is very
# likely not covered by the documents at all, so we refuse rather than
# risk the LLM answering from its own (unverified) general knowledge.
# Calibrated empirically on this corpus: on-topic questions score ~0.45-0.7,
# unrelated questions score <0.3.
MIN_SCORE = 0.35

DONT_KNOW_MESSAGE = (
    "I don't know — none of the retrieved passages are relevant enough "
    "to answer that from these documents."
)

# LLM provider config (Groq's OpenAI-compatible API).
AI_API_KEY = os.environ.get("AI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-oss-120b")
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
        evidence = "\n\n".join(
            f"[{h['source']}]\n{h['text']}" for h in hits
        )
        return f"""Answer the question using ONLY the context below. 
Format your answer as:
1. The answer (be specific and accurate)
2. A "Sources:" section listing which document(s) each part came from

If the context does not contain the answer, say "I don't know" instead of hallucinating.

Context:
{evidence}

Question: {question}
Answer:""".strip()

    @staticmethod
    def _synthesize_answer(question: str, hits: list[dict]) -> str:
        """
        Generate a simple answer from context when no LLM is available.
        This extracts relevant sentences and cites sources.
        """
        if not hits:
            return DONT_KNOW_MESSAGE

        source_texts = []
        for hit in hits:
            source_texts.append(f"[{hit['source']}]\n{hit['text']}")

        synthesized = f"From the retrieved documents:\n\n" + "\n\n".join(source_texts)
        return synthesized + f"\n\nSources: {', '.join(h['source'] for h in hits)}"

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

    def answer(
        self, question: str, top_k: int = TOP_K, min_score: float = MIN_SCORE
    ) -> dict:
        """
        The full RAG flow: retrieve evidence, then generate an answer.

        Returns a dict with the answer, the evidence used, and whether the
        question was considered "grounded" (i.e. answerable from the
        documents) so callers can show retrieval debugging info.
        """
        # 1) Find relevant evidence in the vector store.
        hits = self._retrieve(question, top_k=top_k)

        # 2) Groundedness check: if even the best match is a weak match,
        # don't hand it to the LLM at all. This is what makes the "I don't
        # know" behaviour reliable — it doesn't depend on the LLM choosing
        # to follow the system prompt's instructions.
        best_score = hits[0]["score"] if hits else 0.0
        if best_score < min_score:
            return {
                "answer": DONT_KNOW_MESSAGE,
                "evidence": hits,
                "grounded": False,
            }

        # 3) Turn evidence + question into a prompt.
        prompt = self._build_prompt(question, hits)

        # 4) Try to generate with an LLM; degrade gracefully to raw context.
        generated = self._generate_with_llm(prompt)
        if generated is None:
            generated = self._synthesize_answer(question, hits)

        return {"answer": generated, "evidence": hits, "grounded": True}


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
