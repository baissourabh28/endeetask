"""
Document ingestion pipeline.

Reads plain-text documents, chunks them, embeds each chunk,
and upserts into Endee with metadata for retrieval.

Usage:
    python ingest.py --docs ./docs --index rag_docs
"""

import argparse
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_AUTH = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME = os.getenv("INDEX_NAME", "rag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Chunk settings
CHUNK_SIZE = 400       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks
BATCH_SIZE = 32        # vectors per insert request


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += size - overlap
    return [c for c in chunks if c]


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _read_pdf(path: Path) -> str:
    import fitz  # pymupdf
    text = ""
    with fitz.open(str(path)) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


def load_documents(docs_dir: str) -> list[dict]:
    """
    Load .txt, .pdf, and .docx files from a directory.
    Returns list of {filename, text} dicts.
    """
    readers = {".txt": _read_txt, ".pdf": _read_pdf, ".docx": _read_docx}
    docs = []
    for path in Path(docs_dir).iterdir():
        reader = readers.get(path.suffix.lower())
        if not reader:
            continue
        try:
            text = reader(path)
            if text:
                docs.append({"filename": path.name, "text": text})
        except Exception as e:
            print(f"[Ingest] Skipping {path.name}: {e}")
    print(f"[Ingest] Loaded {len(docs)} documents from {docs_dir}")
    return docs


def ingest(docs_dir: str, index_name: str):
    from embedder import Embedder
    from endee_client import EndeeClient

    client = EndeeClient(ENDEE_URL, ENDEE_AUTH)

    if not client.health():
        raise RuntimeError(f"Endee server not reachable at {ENDEE_URL}")

    embedder = Embedder(EMBED_MODEL)

    # Create index if it doesn't exist
    existing = client.list_indexes()
    full_index_id = f"endee/{index_name}"
    if full_index_id not in existing:
        print(f"[Ingest] Creating index '{index_name}' (dim={embedder.dim})")
        result = client.create_index(
            index_name=index_name,
            dim=embedder.dim,
            space_type="cosine",
            precision="float32",
        )
        print(f"[Ingest] Create index response: {result}")
    else:
        print(f"[Ingest] Index '{index_name}' already exists, skipping creation")

    docs = load_documents(docs_dir)
    if not docs:
        print("[Ingest] No supported documents found (.txt, .pdf, .docx). Add files to the docs directory.")
        return

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": doc["filename"],
                "chunk_index": i,
                "text": chunk,
            })

    print(f"[Ingest] Total chunks to embed: {len(all_chunks)}")

    # Embed and insert in batches
    inserted = 0
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embedder.embed(texts)

        payload = []
        for chunk, vec in zip(batch, vectors):
            payload.append({
                "id": str(uuid.uuid4()),
                "vector": vec,
                "metadata": {
                    "text": chunk["text"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                },
                "filter": {
                    "filename": chunk["filename"],
                },
            })

        result = client.insert_vectors(index_name, payload)
        if result["status"] not in (200, 201):
            print(f"[Ingest] Insert failed: {result}")
        else:
            inserted += len(batch)
            print(f"[Ingest] Inserted {inserted}/{len(all_chunks)} chunks")

    print(f"[Ingest] Done. {inserted} chunks stored in index '{index_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Endee")
    parser.add_argument("--docs", default="./docs", help="Directory with .txt, .pdf, or .docx files")
    parser.add_argument("--index", default=INDEX_NAME, help="Endee index name")
    args = parser.parse_args()
    ingest(args.docs, args.index)
