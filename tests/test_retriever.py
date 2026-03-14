"""
Tests for Retriever — mocks both Embedder and EndeeClient.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch


def _inject_mock_embedder(dim=384):
    """Inject a fake embedder module so lazy imports inside Retriever.__init__ work."""
    mock_emb_instance = MagicMock()
    mock_emb_instance.dim = dim
    mock_emb_instance.embed_one.return_value = [0.1] * dim

    MockEmbedderClass = MagicMock(return_value=mock_emb_instance)
    fake_embedder_module = MagicMock()
    fake_embedder_module.Embedder = MockEmbedderClass
    return fake_embedder_module, mock_emb_instance


class TestRetriever(unittest.TestCase):

    def _make_retriever(self):
        fake_emb_mod, mock_emb = _inject_mock_embedder()

        with patch.dict(sys.modules, {"embedder": fake_emb_mod}), \
             patch("retriever.EndeeClient") as MockClient:

            sys.modules.pop("retriever", None)
            from retriever import Retriever

            mock_client = MagicMock()
            MockClient.return_value = mock_client

            r = Retriever(index_name="test_index", top_k=3)
            r.embedder = mock_emb   # replace with our controlled mock
            r.client = mock_client
        return r, mock_emb, mock_client

    def test_retrieve_returns_chunks(self):
        r, mock_emb, mock_client = self._make_retriever()
        mock_client.search.return_value = [
            {"id": "1", "score": 0.9, "metadata": {"text": "chunk text", "filename": "a.txt", "chunk_index": 0}},
            {"id": "2", "score": 0.8, "metadata": {"text": "other chunk", "filename": "b.txt", "chunk_index": 1}},
        ]
        chunks = r.retrieve("what is RAG?")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "chunk text")
        self.assertEqual(chunks[0]["score"], 0.9)

    def test_retrieve_calls_embed_once(self):
        r, mock_emb, mock_client = self._make_retriever()
        mock_client.search.return_value = []
        r.retrieve("test query")
        mock_emb.embed_one.assert_called_once_with("test query")

    def test_retrieve_with_filename_filter(self):
        r, mock_emb, mock_client = self._make_retriever()
        mock_client.search.return_value = []
        r.retrieve("query", filename_filter="doc.txt")
        call_kwargs = mock_client.search.call_args[1]
        self.assertIsNotNone(call_kwargs.get("filters"))
        self.assertEqual(call_kwargs["filters"][0]["filename"]["$eq"], "doc.txt")

    def test_retrieve_no_filter_by_default(self):
        r, mock_emb, mock_client = self._make_retriever()
        mock_client.search.return_value = []
        r.retrieve("query")
        call_kwargs = mock_client.search.call_args[1]
        self.assertIsNone(call_kwargs.get("filters"))

    def test_retrieve_empty_results(self):
        r, mock_emb, mock_client = self._make_retriever()
        mock_client.search.return_value = []
        chunks = r.retrieve("unknown topic")
        self.assertEqual(chunks, [])

    def test_retrieve_top_k_override(self):
        r, mock_emb, mock_client = self._make_retriever()
        mock_client.search.return_value = []
        r.retrieve("query", top_k=10)
        call_kwargs = mock_client.search.call_args[1]
        self.assertEqual(call_kwargs["k"], 10)


if __name__ == "__main__":
    unittest.main()
