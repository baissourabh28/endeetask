"""
Tests for the generator module.
"""

import unittest
from unittest.mock import patch, MagicMock


class TestGenerator(unittest.TestCase):

    def _chunks(self, n=2):
        return [
            {"text": f"Context chunk {i}", "filename": f"doc{i}.txt", "chunk_index": i, "score": 0.9 - i * 0.1}
            for i in range(n)
        ]

    def test_no_chunks_returns_fallback_message(self):
        from generator import generate_answer
        answer = generate_answer("What is X?", [])
        self.assertIn("No relevant context", answer)

    def test_local_fallback_used_when_no_keys(self):
        with patch("generator.OPENAI_API_KEY", ""), patch("generator.OLLAMA_MODEL", ""):
            from generator import generate_answer
            answer = generate_answer("What is X?", self._chunks())
            self.assertIn("Based on your documents", answer)

    def test_local_fallback_includes_top_chunk(self):
        with patch("generator.OPENAI_API_KEY", ""), patch("generator.OLLAMA_MODEL", ""):
            from generator import generate_answer
            answer = generate_answer("What is X?", self._chunks(2))
            self.assertIn("Context chunk 0", answer)

    def test_local_fallback_combines_top_3(self):
        with patch("generator.OPENAI_API_KEY", ""), patch("generator.OLLAMA_MODEL", ""):
            from generator import generate_answer
            answer = generate_answer("What is X?", self._chunks(5))
            self.assertIn("Context chunk 0", answer)
            self.assertIn("Context chunk 1", answer)
            self.assertIn("Context chunk 2", answer)

    def test_prompt_contains_question(self):
        from generator import _build_prompt
        prompt = _build_prompt("What is HNSW?", self._chunks(1))
        self.assertIn("What is HNSW?", prompt)

    def test_prompt_contains_context(self):
        from generator import _build_prompt
        prompt = _build_prompt("question", self._chunks(2))
        self.assertIn("Context chunk 0", prompt)
        self.assertIn("Context chunk 1", prompt)

    def test_prompt_contains_source_info(self):
        from generator import _build_prompt
        prompt = _build_prompt("question", self._chunks(1))
        self.assertIn("doc0.txt", prompt)

    @patch("generator.OLLAMA_MODEL", "llama3.2")
    @patch("generator.OPENAI_API_KEY", "")
    @patch("generator._call_ollama")
    def test_ollama_called_when_model_set(self, mock_ollama):
        mock_ollama.return_value = "Ollama answer"
        from generator import generate_answer
        answer = generate_answer("What is X?", self._chunks())
        mock_ollama.assert_called_once()
        self.assertEqual(answer, "Ollama answer")

    @patch("generator.OLLAMA_MODEL", "llama3.2")
    @patch("generator.OPENAI_API_KEY", "")
    def test_ollama_connection_error_falls_back(self):
        import requests as req
        with patch("generator.requests.post", side_effect=req.exceptions.ConnectionError):
            from generator import generate_answer
            answer = generate_answer("What is X?", self._chunks())
            self.assertIn("Based on your documents", answer)

    @patch("generator.OLLAMA_MODEL", "")
    @patch("generator.OPENAI_API_KEY", "fake-key")
    @patch("generator._call_openai")
    def test_openai_called_when_key_present_and_no_ollama(self, mock_openai):
        mock_openai.return_value = "OpenAI answer"
        from generator import generate_answer
        answer = generate_answer("What is X?", self._chunks())
        mock_openai.assert_called_once()
        self.assertEqual(answer, "OpenAI answer")

    @patch("generator.OLLAMA_MODEL", "")
    @patch("generator.OPENAI_API_KEY", "fake-key")
    @patch("generator._call_openai")
    def test_openai_error_returns_error_string(self, mock_openai):
        mock_openai.return_value = "[OpenAI error] timeout"
        from generator import generate_answer
        answer = generate_answer("What is X?", self._chunks())
        self.assertIn("OpenAI error", answer)

    @patch("generator.OLLAMA_MODEL", "llama3.2")
    def test_ollama_success_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "  Ollama says hello  "}}
        with patch("generator.requests.post", return_value=mock_resp):
            from generator import _call_ollama
            result = _call_ollama("What is X?", self._chunks())
            self.assertEqual(result, "Ollama says hello")

    @patch("generator.OLLAMA_MODEL", "llama3.2")
    def test_ollama_http_error_returns_error_string(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal server error"
        with patch("generator.requests.post", return_value=mock_resp):
            from generator import _call_ollama
            result = _call_ollama("What is X?", self._chunks())
            self.assertIn("Ollama error", result)


if __name__ == "__main__":
    unittest.main()
