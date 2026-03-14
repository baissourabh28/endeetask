"""
Embedding layer using sentence-transformers (local, no API key needed).
Swap out get_embeddings() to use OpenAI or any other provider.
"""

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Lazy import so tests can mock without the package installed
        from sentence_transformers import SentenceTransformer
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Embedding dimension: {self.dim}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
