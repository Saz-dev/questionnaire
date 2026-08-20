import sys

from chunker import chunk_documents
from debugger import print_inspection, visualize_retrieval
from embedder import embed_texts
from loader import load_pdfs
from rag import RAG
from vectordb import VectorDB

EXIT_COMMANDS = {"exit", "quit", "q", ":q"}

# Retrieval strategies (see rag.py). Default is the week-4 improved
# pipeline: semantic search + cross-encoder reranking.
MODES = ("semantic", "hybrid", "rerank", "hybrid_rerank", "mmr", "hybrid_mmr", "rewrite", "hyde")


def build_index() -> VectorDB:
    """Ingest the PDFs once and return a ready-to-query vector store."""
    print("1/3 Loading documents...")
    documents = load_pdfs()
    print(f"   -> {len(documents)} document(s)")
    if not documents:
        print("   !! No PDFs found in documents/ — add some and re-run.")

    print("2/3 Chunking text...")
    chunks = chunk_documents(documents, chunk_size=500, overlap=100)
    print(f"   -> {len(chunks)} chunks created")

    print("3/3 Embedding & storing vectors...")
    vectors = embed_texts([c["text"] for c in chunks])
    db = VectorDB()
    db.add_chunks(chunks, vectors)
    print(f"   -> {len(chunks)} vectors stored in Qdrant (in-memory)\n")
    return db


def chat(rag: RAG, inspect: bool = False) -> None:
    """Interactive terminal chat loop: ask questions, get grounded answers."""
    print(f"Ask a question about your documents. Type 'exit' to quit.")
    print(f"Retrieval mode: {rag.mode}" + (" | --inspect shows retrieval internals" if inspect else ""))
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Bye.")
            break

        result = rag.answer(question)
        print(f"\nAssistant: {result['answer']}\n")

        if inspect:
            print_inspection(
                visualize_retrieval(rag.db, question, mode=rag.mode)
            )
        else:
            if result["grounded"]:
                print("Sources:")
                for hit in result["evidence"]:
                    score = hit.get("cosine", hit.get("score"))
                    print(f"  - [{hit['source']}] score={score:.3f} "
                          f"-> {hit['text'][:90]}...")
            else:
                print("(No sources met the relevance threshold — nothing to cite.)")
        print()


def main() -> None:
    args = sys.argv[1:]
    mode = "rerank"
    inspect = False
    if args:
        for arg in args:
            if arg.startswith("--mode="):
                mode = arg.split("=", 1)[1]
            elif arg == "--inspect":
                inspect = True
            elif arg in MODES:
                mode = arg
            else:
                print(f"Unknown argument: {arg}")
                print("Usage: python main.py [--mode=NAME] [--inspect]")
                print(f"Modes: {', '.join(MODES)}")
                return
    if mode not in MODES:
        print(f"Unknown mode {mode!r}. Modes: {', '.join(MODES)}")
        return

    db = build_index()
    rag = RAG(db, mode=mode)
    chat(rag, inspect=inspect)


if __name__ == "__main__":
    main()
