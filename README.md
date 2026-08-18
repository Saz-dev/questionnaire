# insurance-queries

An interactive terminal chatbot that answers questions about Honda bike maintenance using Retrieval-Augmented Generation (RAG). Drop your PDFs into `documents/`, ask a question, and get grounded answers drawn from your own documents.

## How it works

The pipeline is split into five small, self-contained steps (each runnable standalone):

| Step | File | What it does |
|------|------|--------------|
| 1. Load | `loader.py` | Extracts text from every PDF in `documents/` |
| 2. Chunk | `chunker.py` | Splits long text into overlapping chunks (configurable size and overlap) |
| 3. Embed | `embedder.py` | Converts chunks into 384-dim vectors via `all-MiniLM-L6-v2` |
| 4. Store & retrieve | `vectordb.py` | Stores vectors in an in-memory Qdrant collection and answers cosine-similarity queries |
| 5. Generate | `rag.py` | Retrieves the top-k most relevant chunks and feeds them to an LLM to generate a grounded answer |

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
python main.py
```

The script ingests the PDFs (load → chunk → embed → store) and starts an interactive prompt:

```
Ask a question about your documents. Type 'exit' to quit.

You: What documents are needed for a total loss claim?
Assistant: To file a total loss claim you need...
Sources:
  - [owner-manual.pdf] score=0.812 -> ...
```

Each answer includes its sources and similarity scores so you can audit where the answer came from. Type `exit`, `quit`, or `q` to leave.

### Running steps individually

Each module has a self-contained `__main__` test:

```bash
python loader.py            # load and print PDF text lengths
python chunker.py           # quick chunking sanity check
python embedder.py          # embedding similarity check
python vectordb.py          # store + retrieve a test chunk
python rag.py               # end-to-end answer on a sample question
```

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
   filenames + similarity scores used) and a `grounded` boolean, so callers
   can always show *why* an answer was given — or why it wasn't.

## Notes

- The vector store is in-memory (`QdrantClient(":memory:")`), so the index is rebuilt on every run. To persist it, point `vectordb.py` at a Qdrant server URL.
- Chunk size and overlap are configurable in `main.py` (`chunk_size=500, overlap=100`) — smaller chunks may lose context, while larger chunks dilute the embedding without improving recall further. Test different configurations with your own documents to find the optimal trade-off.
- The LLM is instructed to answer *only* from the retrieved context — if the context doesn't contain the answer, it says so rather than hallucinating. This is now backed by the similarity-threshold refusal in `rag.py` (see above), not just the prompt instruction.