from pathlib import Path

from pypdf import PdfReader


def load_pdfs(documents_dir: str = "documents") -> list[dict]:
    """
    Extract text from every PDF file in a folder.

    Args:
        documents_dir: relative path to the folder holding the PDFs.

    Returns:
        A list of documents, each a dict like:
            {"text": "...extracted text...", "source": "name.pdf"}
    """
    documents = []

    # Path.glob lets us enumerate files matching a pattern inside a folder.
    for pdf_path in Path(documents_dir).glob("*.pdf"):
        # pypdf's PdfReader gives us access to each page of the PDF.
        reader = PdfReader(str(pdf_path))

        # Concatenate the text of every page into one big string.
        # PDFs are layout files (not "text" files), so page.extract_text()
        # is an OCR-ish best-effort conversion and can be a bit messy.
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        # Strip whitespace and keep the filename as provenance metadata.
        documents.append({"text": text.strip(), "source": pdf_path.name})

    return documents


if __name__ == "__main__":
    # Quick manual test: python loader.py
    docs = load_pdfs()
    for doc in docs:
        print(f"Loaded {doc['source']}: {len(doc['text'])} characters")
