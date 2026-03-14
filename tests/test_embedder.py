"""
Tests for Embedder.
Patches sentence_transformers at the sys.modules level so the lazy
import inside __init__ picks up the mock without the package installed.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


def _make_mock_st_module(dim: int):
    """Return a fake sentence_transformers module with a mock SentenceTransformer class."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim

    MockST = MagicMock(return_value=mock_model)

    fake_module = MagicMock()
    fake_module.SentenceTransformer = MockST
    return fake_module, mock_model


class TestEmbedder(unittest.TestCase):

    def _make_embedder(self, dim=384):
        fake_module, mock_model = _make_mock_st_module(dim)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            # Remove cached embedder module so it re-imports with the mock
            sys.modules.pop("embedder", None)
            from embedder import Embedder
            emb = Embedder("all-MiniLM-L6-v2")
        emb.model = mock_model
        emb.dim = dim
        return emb, mock_model, dim

    def test_dim_set_correctly(self):
        emb, _, dim = self._make_embedder(384)
        self.assertEqual(emb.dim, 384)

    def test_embed_returns_list_of_lists(self):
        emb, mock_model, dim = self._make_embedder(384)
        mock_model.encode.return_value = np.random.rand(2, dim).astype(np.float32)
        result = emb.embed(["hello", "world"])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), dim)

    def test_embed_one_returns_single_vector(self):
        emb, mock_model, dim = self._make_embedder(384)
        mock_model.encode.return_value = np.random.rand(1, dim).astype(np.float32)
        result = emb.embed_one("test sentence")
        self.assertEqual(len(result), dim)
        self.assertIsInstance(result, list)

    def test_embed_empty_list(self):
        emb, mock_model, dim = self._make_embedder(384)
        mock_model.encode.return_value = np.array([]).reshape(0, dim).astype(np.float32)
        result = emb.embed([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
