"""
Integration-style tests for RAGPipeline.
Mocks Retriever and generate_answer so no server or model is needed.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch


def _make_pipeline():
    # Stub out the heavy deps so rag_pipeline imports cleanly
    fake_retriever_instance = MagicMock()
    MockRetrieverClass = MagicMock(return_value=fake_retriever_instance)

    fake_retriever_mod = MagicMock()
    fake_retriever_mod.Retriever = MockRetrieverClass

    fake_gen_mod = MagicMock()
    mock_gen = MagicMock(return_value="mocked answer")
    fake_gen_mod.generate_answer = mock_gen

    # Also stub embedder so retriever import chain doesn't blow up
    fake_emb_mod = MagicMock()
    fake_emb_mod.Embedder = MagicMock()

    with patch.dict(sys.modules, {
        "retriever": fake_retriever_mod,
        "generator": fake_gen_mod,
        "embedder": fake_emb_mod,
    }):
        sys.modules.pop("rag_pipeline", None)
        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(index_name="test", top_k=3)
        pipeline.retriever = fake_retriever_instance

    return pipeline, fake_retriever_instance, mock_gen


class TestRAGPipeline(unittest.TestCase):

    def test_ask_returns_expected_keys(self):
        pipeline, mock_retriever, mock_gen = _make_pipeline()
        mock_retriever.retrieve.return_value = [
            {"text": "chunk", "filename": "a.txt", "chunk_index": 0, "score": 0.9, "id": "1"}
        ]
        mock_gen.return_value = "The answer"

        result = pipeline.ask("What is RAG?")
        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("sources", result)

    def test_ask_passes_question_to_retriever(self):
        pipeline, mock_retriever, mock_gen = _make_pipeline()
        mock_retriever.retrieve.return_value = []
        mock_gen.return_value = "no context"

        pipeline.ask("What is HNSW?")
        mock_retriever.retrieve.assert_called_once_with("What is HNSW?", filename_filter=None)

    def test_ask_passes_filename_filter(self):
        pipeline, mock_retriever, mock_gen = _make_pipeline()
        mock_retriever.retrieve.return_value = []
        mock_gen.return_value = "no context"

        pipeline.ask("question", filename_filter="doc.txt")
        mock_retriever.retrieve.assert_called_once_with("question", filename_filter="doc.txt")

    def test_sources_truncated_to_200_chars(self):
        pipeline, mock_retriever, mock_gen = _make_pipeline()
        long_text = "X" * 500
        mock_retriever.retrieve.return_value = [
            {"text": long_text, "filename": "a.txt", "chunk_index": 0, "score": 0.9, "id": "1"}
        ]
        mock_gen.return_value = "answer"

        result = pipeline.ask("q")
        self.assertLessEqual(len(result["sources"][0]["text"]), 204)  # 200 + "..."

    def test_empty_retrieval_still_returns_answer(self):
        pipeline, mock_retriever, mock_gen = _make_pipeline()
        mock_retriever.retrieve.return_value = []
        mock_gen.return_value = "No relevant context found in the knowledge base."

        result = pipeline.ask("unknown question")
        self.assertEqual(result["sources"], [])
        self.assertIn("No relevant context", result["answer"])

    def test_question_echoed_in_result(self):
        pipeline, mock_retriever, mock_gen = _make_pipeline()
        mock_retriever.retrieve.return_value = []
        mock_gen.return_value = "answer"

        result = pipeline.ask("Is Endee fast?")
        self.assertEqual(result["question"], "Is Endee fast?")


if __name__ == "__main__":
    unittest.main()
