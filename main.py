from chunker import chunk_documents
from embedder import embed_texts
from loader import load_pdfs
from rag import RAG
from vectordb import VectorDB

EXIT_COMMANDS = {"exit", "quit", "q", ":q"}


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


def chat(rag: RAG) -> None:
    """Interactive terminal chat loop: ask questions, get grounded answers."""
    print("Ask a question about your documents. Type 'exit' to quit.\n")

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

        print("Sources:")
        for hit in result["evidence"]:
            print(f"  - [{hit['source']}] score={hit['score']:.3f} "
                  f"-> {hit['text'][:90]}...")
        print()


def main() -> None:
    db = build_index()
    rag = RAG(db)
    chat(rag)


if __name__ == "__main__":
    main()
