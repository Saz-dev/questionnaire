import os

from dotenv import load_dotenv

from embedder import embed_text
from queryrewrite import hyde, rewrite_llm
from reranker import mmr_rerank, rerank
from vectordb import VectorDB

# Load AI_API_KEY / AI_MODEL / APP_NAME from .env (no-op if not present).
load_dotenv()

# How many chunks to feed the LLM as "evidence".
TOP_K = 3

# How many candidates the cheap first pass returns before reranking.
CANDIDATES = 50

# Minimum cosine similarity a retrieved chunk needs before we trust it
# enough to generate an answer from it. Below this, the question is very
# likely not covered by the documents at all, so we refuse rather than
# risk the LLM answering from its own (unverified) general knowledge.
# Calibrated empirically on this corpus: on-topic questions score ~0.45-0.7,
# unrelated questions score <0.3.
MIN_SCORE = 0.35

# Bump this string any time _build_prompt()'s template changes, so old
# traces can be told apart from traces produced by a different prompt.
PROMPT_VERSION = "v1-2026-08-26"

DONT_KNOW_MESSAGE = (
    "I don't know — none of the retrieved passages are relevant enough "
    "to answer that from these documents."
)

DEFAULT_MODE = "rerank"

# LLM provider config (Groq's OpenAI-compatible API).
AI_API_KEY = os.environ.get("AI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-oss-120b")
AI_BASE_URL = "https://api.groq.com/openai/v1"
AI_APP_NAME = os.environ.get("APP_NAME", "Smart CLI")


class RAG:
    """End-to-end retrieval-augmented generation over one VectorDB."""

    def __init__(self, db: VectorDB, mode: str = DEFAULT_MODE):
        self.db = db
        self.mode = mode

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        mode: str | None = None,
        candidates: int = CANDIDATES,
        mmr_lambda: float = 0.7,
    ) -> tuple[list[dict], dict]:
        """
        Retrieve evidence with a chosen strategy.

        Returns (hits, meta) where meta explains HOW the hit list was
        produced — mode name, candidate pool size, rewritten query /
        hypothetical passage when applicable — so the inspection view can
        show exactly what happened.
        """
        mode = mode or self.mode
        meta = {"mode": mode}

        if mode == "semantic":
            hits = self.db.search(embed_text(question), top_k=top_k)

        elif mode == "hybrid":
            hits = self.db.hybrid_search(embed_text(question), question, top_k=top_k)

        elif mode == "rerank":
            pool = self.db.search(embed_text(question), top_k=candidates)
            meta["candidate_pool"] = pool
            hits = rerank(question, pool, top_k=top_k)

        elif mode == "mmr":
            pool = self.db.search(embed_text(question), top_k=candidates)
            meta["candidate_pool"] = pool
            hits = mmr_rerank(embed_text(question), pool, top_k=top_k, lambda_=mmr_lambda)

        elif mode == "hybrid_mmr":
            # Bonus (Task Set D §5): MMR over the FUSED (BM25+RRF) candidate
            # list, not just semantic — so the diversity pass also sees
            # keyword-only hits before it starts discounting near-duplicates.
            pool = self.db.hybrid_search(embed_text(question), question, top_k=candidates)
            meta["candidate_pool"] = pool
            hits = mmr_rerank(embed_text(question), pool, top_k=top_k, lambda_=mmr_lambda)

        elif mode == "hybrid_rerank":
            pool = self.db.hybrid_search(embed_text(question), question, top_k=candidates)
            meta["candidate_pool"] = pool
            hits = rerank(question, pool, top_k=top_k)

        elif mode == "rewrite":
            rewritten = rewrite_llm(question)
            meta["rewritten"] = rewritten
            hits = self.db.search(embed_text(rewritten), top_k=top_k)

        elif mode == "hyde":
            passage = hyde(question)
            meta["hypothetical"] = passage
            hits = self.db.search(embed_text(passage), top_k=top_k)

        else:
            raise ValueError(f"Unknown retrieval mode: {mode!r}")

        return hits, meta

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
        for attempt in range(8):
            try:
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
                    temperature=0.2,  # low temp -> factual, reproducible answers
                )
                return response.choices[0].message.content
            except Exception as exc:
                if getattr(exc, "status_code", None) in (429, 503):
                    import time

                    time.sleep(min(30 * (2 ** attempt), 240))  # rate limit: back off
                else:
                    print(f"  ! LLM generation failed: {exc}")
                    return None
        return None

    # ------------------------------------------------------------------
    # End-to-end answer
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
        mode: str | None = None,
        mmr_lambda: float = 0.7,
    ) -> dict:
        """
        The full RAG flow: retrieve evidence, then generate an answer.

        Returns a dict with the answer, the evidence used, retrieval meta
        (mode, candidates, rewrites) and whether the question was
        considered "grounded" (i.e. answerable from the documents) so
        callers can show retrieval debugging info.
        """
        # 1) Find relevant evidence in the vector store.
        hits, meta = self.retrieve(question, top_k=top_k, mode=mode, mmr_lambda=mmr_lambda)

        # 2) Groundedness check: if even the best candidate is a weak
        # match, don't hand it to the LLM at all. Uses the RAW cosine of
        # the candidate pool (not rerank scores, which are on a
        # different scale).
        pool = meta.get("candidate_pool") or hits
        best_cosine = max(
            (h.get("cosine") for h in pool if h.get("cosine") is not None),
            default=0.0,
        )
        if best_cosine < min_score:
            return {
                "answer": DONT_KNOW_MESSAGE,
                "evidence": hits,
                "meta": meta,
                "grounded": False,
            }

        # 3) Turn evidence + question into a prompt.
        prompt = self._build_prompt(question, hits)

        # 4) Try to generate with an LLM; degrade gracefully to raw context.
        generated = self._generate_with_llm(prompt)
        if generated is None:
            generated = self._synthesize_answer(question, hits)

        return {"answer": generated, "evidence": hits, "meta": meta, "grounded": True}


if __name__ == "__main__":
    # Quick manual test: python rag.py
    from loader import load_pdfs
    from chunker import chunk_documents
    from embedder import embed_texts

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    db = VectorDB()
    db.add_chunks(chunks, embed_texts([c["text"] for c in chunks]))

    result = RAG(db).answer("What brake fluid does the bike use?")
    print("ANSWER:", result["answer"][:400])
