"""
RAG Pipeline: ties retriever + generator together.
Can be used as a library or run directly as a CLI.

Usage:
    python rag_pipeline.py --query "What is HNSW?"
    python rag_pipeline.py --query "Explain sparse search" --file sparse.txt
    python rag_pipeline.py --interactive
    python rag_pipeline.py --upload          # asks for file path, ingests, then Q&A
"""

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from retriever import Retriever

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME", "rag_docs")
TOP_K = int(os.getenv("TOP_K", "5"))
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


class RAGPipeline:
    def __init__(self, index_name: str = INDEX_NAME, top_k: int = TOP_K):
        self.retriever = Retriever(index_name=index_name, top_k=top_k)

    def ask(self, question: str, filename_filter: str = None) -> dict:
        """
        Full RAG cycle: retrieve → generate.

        Returns:
            {
                "question": str,
                "answer": str,
                "sources": [{"filename", "chunk_index", "score", "text"}, ...]
            }
        """
        chunks = self.retriever.retrieve(question, filename_filter=filename_filter)
        from generator import generate_answer
        answer = generate_answer(question, chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "filename": c["filename"],
                    "chunk_index": c["chunk_index"],
                    "score": round(c["score"], 4),
                    "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                }
                for c in chunks
            ],
        }


def _print_result(result: dict):
    print("\n" + "=" * 60)
    print(f"Question: {result['question']}")
    print("=" * 60)
    print(f"\nAnswer:\n{result['answer']}")
    print("\nSources:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['filename']} (chunk {s['chunk_index']}, score={s['score']})")
        print(f"      {s['text'][:120]}...")
    print()


def _upload_and_ingest(index_name: str):
    """Ask user for a file path, copy it to docs/, and re-ingest into Endee."""
    print("\n--- Upload a Document ---")
    print("Supported formats: .txt  .pdf  .docx")
    print("Tip: drag and drop the file into this terminal to paste its path\n")

    while True:
        raw = input("Enter file path: ").strip().strip('"').strip("'")
        if not raw:
            print("No path entered. Skipping upload.")
            return None
        src = Path(raw)
        if not src.exists():
            print(f"  File not found: {src}  — try again")
            continue
        if src.suffix.lower() not in (".txt", ".pdf", ".docx"):
            print(f"  Unsupported format '{src.suffix}' — only .txt, .pdf, .docx allowed")
            continue
        break

    # Copy to docs/ only if not already there
    dest = Path(DOCS_DIR) / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
        print(f"\n  Copied '{src.name}' → docs/")
    else:
        print(f"\n  '{src.name}' is already in docs/ — skipping copy")

    # Re-ingest
    print(f"  Ingesting '{src.name}' into Endee index '{index_name}'...\n")
    from ingest import ingest
    ingest(DOCS_DIR, index_name)
    print(f"\n  Done. You can now ask questions about '{src.name}'")
    return src.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline using Endee")
    parser.add_argument("--query", "-q", type=str, help="Single question to answer")
    parser.add_argument("--file", "-f", type=str, help="Filter results to a specific filename")
    parser.add_argument("--index", type=str, default=INDEX_NAME, help="Endee index name")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--upload", "-u", action="store_true", help="Upload a file then start interactive Q&A")
    args = parser.parse_args()

    # --upload: ask for file, ingest, then drop into interactive mode
    if args.upload:
        uploaded = _upload_and_ingest(args.index)
        pipeline = RAGPipeline(index_name=args.index, top_k=args.top_k)
        print("\nRAG Pipeline — Interactive Mode (type 'exit' to quit)")
        filter_on = uploaded  # auto-filter to the uploaded file
        while True:
            try:
                q = input("\nYour question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            result = pipeline.ask(q, filename_filter=filter_on)
            _print_result(result)

    elif args.interactive:
        pipeline = RAGPipeline(index_name=args.index, top_k=args.top_k)
        print("RAG Pipeline — Interactive Mode (type 'exit' to quit)")
        print("Tip: run with --upload first to add your own document\n")
        while True:
            try:
                q = input("\nYour question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            result = pipeline.ask(q, filename_filter=args.file)
            _print_result(result)

    elif args.query:
        pipeline = RAGPipeline(index_name=args.index, top_k=args.top_k)
        result = pipeline.ask(args.query, filename_filter=args.file)
        _print_result(result)

    else:
        parser.print_help()

