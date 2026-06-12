"""
Knowledge Base indexer: chunk → embed → upsert (Pinecone) and delete-by-document.

This is the UNSTRUCTURED-data path (PDF / TXT / CSV-as-text / pasted policies &
FAQs). Each Knowledge Document owned by the Next.js dashboard is split into
chunks, embedded with the same Gemini embedding model used everywhere else, and
upserted into Pinecone with a stable id prefix (``kb_<document_id>_<n>``) plus
``documentId`` metadata so the whole document can be re-indexed or deleted as a
unit. Structured product data must NOT come through here — it belongs in
PostgreSQL via the Inventory Sync.
"""

from __future__ import annotations

import io

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_sales.config.vectorstore import get_embeddings, get_pinecone_index

_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50
_UPSERT_BATCH = 100


def _vector_id(document_id: str, index: int) -> str:
    return f"kb_{document_id}_{index}"


def extract_text(data: bytes, filename: str) -> str:
    """Extract plain text from an uploaded file (pdf / txt / csv)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()

    # txt / csv / md / anything text-like: decode as UTF-8 (best effort).
    return data.decode("utf-8", errors="replace").strip()


def _chunk(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def index_document(document_id: str, title: str, text: str) -> int:
    """Chunk, embed and upsert a document's text into Pinecone.

    Re-indexing is safe: existing chunks for this document are removed first so
    stale chunks never linger after an edit.

    Returns:
        The number of chunks (vectors) stored.
    """
    if not document_id:
        raise ValueError("document_id is required")

    # Clear any previous chunks for this document before re-indexing.
    delete_document(document_id)

    chunks = _chunk(text or "")
    if not chunks:
        return 0

    embeddings = get_embeddings()
    index = get_pinecone_index()

    stored = 0
    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start : start + _UPSERT_BATCH]
        vectors = embeddings.embed_documents(batch)

        records = []
        for offset, (chunk_text, vector) in enumerate(zip(batch, vectors)):
            records.append(
                {
                    "id": _vector_id(document_id, start + offset),
                    "values": vector,
                    "metadata": {
                        "source_type": "knowledge",
                        "documentId": document_id,
                        "title": title,
                        "text": chunk_text,
                    },
                }
            )
        index.upsert(vectors=records)
        stored += len(records)

    return stored


def delete_document(document_id: str) -> int:
    """Delete all Pinecone vectors belonging to a document.

    Uses id-prefix listing (supported on serverless) so we never touch other
    documents' vectors. Returns the number of ids requested for deletion.
    """
    if not document_id:
        return 0

    index = get_pinecone_index()
    prefix = f"kb_{document_id}_"

    ids: list[str] = []
    try:
        # `index.list` yields pages of ids matching the prefix (serverless).
        for page in index.list(prefix=prefix):
            if isinstance(page, list):
                ids.extend(page)
            else:
                ids.append(page)
    except Exception:
        # Older clients / pod indexes may not support prefix listing. Best
        # effort: nothing to delete deterministically, so bail out quietly.
        ids = []

    if ids:
        # Delete in batches to stay within request limits.
        for start in range(0, len(ids), 1000):
            index.delete(ids=ids[start : start + 1000])

    return len(ids)
