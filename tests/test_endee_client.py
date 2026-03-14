"""
Tests for EndeeClient.
Uses unittest.mock to avoid needing a live server.
"""

import unittest
from unittest.mock import patch, MagicMock
from endee_client import EndeeClient


class TestEndeeClient(unittest.TestCase):

    def setUp(self):
        self.client = EndeeClient("http://localhost:8080", auth_token="test-token")

    @patch("endee_client.requests.get")
    def test_health_ok(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        self.assertTrue(self.client.health())

    @patch("endee_client.requests.get")
    def test_health_fail(self, mock_get):
        mock_get.side_effect = Exception("connection refused")
        self.assertFalse(self.client.health())

    @patch("endee_client.requests.post")
    def test_create_index(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, text="Index created successfully")
        result = self.client.create_index("my_index", dim=384)
        self.assertEqual(result["status"], 200)
        call_payload = mock_post.call_args[1]["json"]
        self.assertEqual(call_payload["dim"], 384)
        self.assertEqual(call_payload["index_name"], "my_index")

    @patch("endee_client.requests.post")
    def test_search_returns_results(self, mock_post):
        import json as _json, msgpack
        # Endee returns msgpack: list of [similarity, id, meta_bytes, filter, norm, vector]
        meta_bytes = _json.dumps({"text": "hello"}).encode("utf-8")
        mock_post.return_value = MagicMock(
            status_code=200,
            content=msgpack.packb([[0.95, "abc", meta_bytes, "", 0.0, []]]),
        )
        results = self.client.search("my_index", vector=[0.1] * 384, k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "abc")

    @patch("endee_client.requests.post")
    def test_search_empty_on_error(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        results = self.client.search("my_index", vector=[0.1] * 384)
        self.assertEqual(results, [])

    @patch("endee_client.requests.post")
    def test_insert_vectors_sends_correct_payload(self, mock_post):
        import json as _json
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        vectors = [{"id": "1", "vector": [0.1] * 384, "metadata": {"text": "doc"}}]
        result = self.client.insert_vectors("my_index", vectors)
        self.assertEqual(result["status"], 200)
        # New API: sends a JSON array via data= (not json=), top-level list
        sent_data = mock_post.call_args[1]["data"]
        sent = _json.loads(sent_data)
        self.assertIsInstance(sent, list)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["id"], "1")

    @patch("endee_client.requests.get")
    def test_list_indexes(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"indexes": ["endee/rag_docs"]},
        )
        indexes = self.client.list_indexes()
        self.assertIn("endee/rag_docs", indexes)

    def test_auth_header_set(self):
        self.assertEqual(self.client.headers["Authorization"], "test-token")

    def test_no_auth_header_when_empty(self):
        c = EndeeClient("http://localhost:8080", auth_token="")
        self.assertNotIn("Authorization", c.headers)

    @patch("endee_client.requests.delete")
    def test_delete_vector_success(self, mock_delete):
        mock_delete.return_value = MagicMock(status_code=204)
        result = self.client.delete_vector("my_index", "vec-123")
        self.assertTrue(result)
        mock_delete.assert_called_once()
        url = mock_delete.call_args[0][0]
        self.assertIn("vec-123", url)

    @patch("endee_client.requests.delete")
    def test_delete_vector_not_found(self, mock_delete):
        mock_delete.return_value = MagicMock(status_code=404)
        result = self.client.delete_vector("my_index", "missing")
        self.assertFalse(result)

    @patch("endee_client.requests.delete")
    def test_delete_index_success(self, mock_delete):
        mock_delete.return_value = MagicMock(status_code=200)
        result = self.client.delete_index("my_index")
        self.assertTrue(result)

    @patch("endee_client.requests.delete")
    def test_delete_index_fail(self, mock_delete):
        mock_delete.return_value = MagicMock(status_code=500)
        result = self.client.delete_index("my_index")
        self.assertFalse(result)

    @patch("endee_client.requests.get")
    def test_index_info_returns_dict(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"dim": 384, "space_type": "cosine", "total_elements": 100},
        )
        info = self.client.index_info("my_index")
        self.assertEqual(info["dim"], 384)

    @patch("endee_client.requests.get")
    def test_index_info_empty_on_error(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        info = self.client.index_info("missing_index")
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()
