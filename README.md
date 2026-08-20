# questionnaire

An interactive terminal chatbot that answers questions about Honda bike maintenance using Retrieval-Augmented Generation (RAG). Drop your PDFs into `documents/`, ask a question, and get grounded answers drawn from your own documents.

## How it works

The pipeline is split into five small, self-contained steps (each runnable standalone):

| Step | File | What it does |
|------|------|--------------|
| 1. Load | `loader.py` | Extracts text from every PDF in `documents/` |
| 2. Chunk | `chunker.py` | Splits long text into overlapping chunks (configurable size and overlap) |
| 3. Embed | `embedder.py` | Converts chunks into 384-dim vectors via `all-MiniLM-L6-v2` |
| 4. Store & retrieve | `vectordb.py` | Stores vectors in an in-memory Qdrant collection; semantic + BM25 + hybrid (RRF) retrieval |
| 5. Generate | `rag.py` | Retrieves evidence (pluggable strategy), reranks it, and feeds it to an LLM to generate a grounded answer |

## Week 4 — Retrieval & RAG debugging

The week's modules, all in this repo:

| Tool | File | What it gives you |
|------|------|-------------------|
| Inspection view | `debugger.py` | Question / what was fetched (semantic, BM25, hybrid, reranked) / final answer, side by side |
| BM25 keyword search | `vectordb.py` | Real Okapi BM25 (k1=1.5, b=0.75) for exact terms (`MR6K-9`, `30 A`, `1.5 mm`) |
| Hybrid search | `vectordb.py` | Semantic + keyword fused with Reciprocal Rank Fusion |
| Cross-encoder reranking | `reranker.py` | Second pass over top-50 candidates (`cross-encoder/ms-marco-MiniLM-L-6-v2`; drop-in for Cohere Rerank / BGE-Reranker) |
| MMR | `reranker.py` | Maximal Marginal Relevance — diversity selection (over semantic OR the fused hybrid pool, `mode="mmr"` / `mode="hybrid_mmr"`) |
| Query rewriting | `queryrewrite.py` | Rule-based + LLM rewriting of messy questions |
| HyDE | `queryrewrite.py` | Hypothetical Document Embeddings — embed a generated answer-passage instead of the question |
| Evaluation | `evaluate.py` | Golden test set, hit-rate@k / recall@k / MRR, failure classification, before/after report (`WEEK4_REPORT.md`) |
| chunk_id tagging | `vectordb.py` | Every chunk gets a stable `source::index` ID (`find_chunk_id`, `corpus_contains`), so a golden set can pin "the correct chunk" instead of just a keyword |
| Task Set D deliverable | `taskd_eval.py` | The graded submission: 12-question golden set with known chunk_ids, R/G/Not-in-Corpus tally, one change, before/after hit-rate@3 + p50 latency, bonus MMR-on-fused-list lambda sweep — writes `golden_set.jsonl` + `results.md` |

The two kinds of wrong:

- **wrong_document** — the right chunk never made it into the evidence. Retrieval bug; a smarter LLM changes nothing.
- **right_document_wrong_answer** — the right chunk was fetched, but the answer still missed the fact. Generation bug.

## Requirements

- Python 3.10+
- Dependencies: `pypdf`, `sentence-transformers`, `qdrant-client`, `python-dotenv`, and optionally `openai` for LLM generation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup

1. **Add your PDFs** to the `documents/` folder (e.g. `documents/owner-manual.pdf`).
2. **Configure the LLM (optional).** Copy the settings into `.env`:

   ```
   AI_API_KEY=your_groq_api_key
   AI_MODEL=llama-3.3-70b-versatile
   APP_NAME=Smart CLI
   ```

   Generation uses Groq's OpenAI-compatible API. Without a key the pipeline still runs fully locally and returns the retrieved context as the answer.

## Usage

```bash
python main.py --mode=rerank --inspect
```

Retrieval strategies (`--mode=`): `semantic` (week-3 baseline), `hybrid` (BM25+RRF), `rerank` (default: semantic + cross-encoder), `hybrid_rerank`, `mmr`, `hybrid_mmr` (MMR over the fused hybrid pool), `rewrite` (LLM query rewrite), `hyde`. `--inspect` prints the full retrieval internals after every answer.

```
Ask a question about your documents. Type 'exit' to quit.
Retrieval mode: rerank

You: What documents are needed for a total loss claim?
Assistant: ...
Sources:
  - [owner-manual.pdf] score=0.812 -> ...
```

### Debugging & evaluation

```bash
python debugger.py inspect "what is the main fuse rating?"   # side-by-side inspection view
python debugger.py inspect "what is the main fuse rating?" --full   # full chunk texts
python debugger.py classify "what is the main fuse rating?" "30 a"  # failure type
python debugger.py metrics          # retrieval metrics over the golden set
python evaluate.py                  # full experiment: baseline vs one change, writes WEEK4_REPORT.md
python evaluate.py metrics hybrid hyde   # metrics for specific modes
python taskd_eval.py                # Task Set D deliverable: writes golden_set.jsonl + results.md
```

`python evaluate.py` measures retrieval quality as a number before and after **one** change — cross-encoder reranking — and writes the report with the failing set, the failure classification, the before/after table, and the failures the change did *not* fix. See `WEEK4_REPORT.md` for the current numbers.

`python taskd_eval.py` is the graded Week 4 Task Set D submission (see `W4-Task-Set-D.md`): a 12-question golden set pinned to known-correct `chunk_id`s, a baseline hit-rate@3, every miss labelled R (retrieval)/G (generation)/Not-in-Corpus with one line of evidence, one chosen change justified by that tally, before/after hit-rate@3 **and** p50 latency, a per-question fixed/unfixed table, and a bonus MMR-over-the-fused-list lambda sweep. See `results.md` and `EXPLANATION.md` for the full walkthrough and current numbers.

## Correctness, citations, and "I don't know"

`rag.py` doesn't rely on the LLM alone to refuse unanswerable questions —
that's fragile, since a model can ignore the system prompt. Instead:

1. Retrieval always runs first, and the **cosine similarity of the best
   match is checked against a threshold** (`MIN_SCORE = 0.35`, calibrated
   against this corpus — on-topic questions score ~0.45-0.7, unrelated ones
   score <0.3). Below the threshold, the pipeline returns a fixed "I don't
   know" message **without calling the LLM at all**, so there's no chance
   of a hallucinated answer to an out-of-scope question.
2. Above the threshold, the retrieved chunks are still passed to the LLM
   with an explicit "use ONLY this context, cite your sources" instruction
   as a second layer of grounding.
3. Every `RAG.answer()` result includes `evidence` (the chunks + source
   filenames + similarity scores used), `meta` (which retrieval strategy
   produced the evidence, candidate pool, rewrites) and a `grounded`
   boolean, so callers can always show *why* an answer was given — or why
   it wasn't.

## Notes

- The vector store is in-memory (`QdrantClient(":memory:")`), so the index is rebuilt on every run. To persist it, point `vectordb.py` at a Qdrant server URL.
- Chunk size and overlap are configurable in `main.py` (`chunk_size=500, overlap=100`) — smaller chunks may lose context, while larger chunks dilute the embedding without improving recall further.
- Reranking scores (cross-encoder logits) are on a different scale than cosine similarity; the groundedness threshold is always applied to the raw candidate cosine, never to rerank scores.
- The LLM is instructed to answer *only* from the retrieved context — if the context doesn't contain the answer, it says so rather than hallucinating. This is now backed by the similarity-threshold refusal in `rag.py` (see above), not just the prompt instruction.

## How to run everything

```bash
source venv/bin/activate

# The Task Set D deliverable (writes golden_set.jsonl + results.md):
python taskd_eval.py

# The general Week 4 toolkit demo (writes WEEK4_REPORT.md, different golden set):
python evaluate.py

# Inspection view for any single question:
python debugger.py inspect "What is the main fuse rating?" semantic
python debugger.py inspect "What is the main fuse rating?" rerank

# Interactive chat with a chosen retrieval mode:
python main.py --mode=hybrid_mmr --inspect
```
