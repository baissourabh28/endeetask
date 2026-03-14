"""
Retriever: embeds a query and fetches top-k relevant chunks from Endee.
Supports optional metadata filters (e.g. restrict to a specific filename).
"""

import os
from typing import Optional

from dotenv import load_dotenv

from endee_client import EndeeClient

load_dotenv()

ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_AUTH = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME = os.getenv("INDEX_NAME", "rag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))


class Retriever:
    def __init__(
        self,
        index_name: str = INDEX_NAME,
        top_k: int = TOP_K,
        embed_model: str = EMBED_MODEL,
    ):
        from embedder import Embedder
        self.client = EndeeClient(ENDEE_URL, ENDEE_AUTH)
        self.embedder = Embedder(embed_model)
        self.index_name = index_name
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filename_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Embed query, search Endee, return list of chunk dicts:
          {id, score, text, filename, chunk_index}

        filename_filter: restrict results to a specific source file.
        """
        k = top_k or self.top_k
        query_vec = self.embedder.embed_one(query)

        filters = None
        if filename_filter:
            filters = [{"filename": {"$eq": filename_filter}}]

        raw_results = self.client.search(
            index_name=self.index_name,
            vector=query_vec,
            k=k,
            filters=filters,
        )

        chunks = []
        for r in raw_results:
            meta = r.get("metadata", {})
            chunks.append({
                "id": r.get("id", ""),
                "score": r.get("score", 0.0),
                "text": meta.get("text", ""),
                "filename": meta.get("filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
            })

        return chunks
