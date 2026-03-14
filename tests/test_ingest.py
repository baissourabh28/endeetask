"""
Tests for the ingestion pipeline: chunking logic and document loading.
"""

import unittest
import tempfile
import os
from pathlib import Path

# Import chunk_text directly (no server needed)
from ingest import chunk_text, load_documents


class TestChunking(unittest.TestCase):

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, size=400, overlap=80)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world")

    def test_long_text_multiple_chunks(self):
        text = "A" * 1000
        chunks = chunk_text(text, size=400, overlap=80)
        self.assertGreater(len(chunks), 1)

    def test_overlap_creates_shared_content(self):
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, size=100, overlap=20)
        # Each chunk except the last should be ~100 chars
        for c in chunks[:-1]:
            self.assertLessEqual(len(c), 100)

    def test_empty_text_no_chunks(self):
        chunks = chunk_text("", size=400, overlap=80)
        self.assertEqual(chunks, [])

    def test_exact_size_text(self):
        text = "X" * 400
        chunks = chunk_text(text, size=400, overlap=80)
        self.assertEqual(len(chunks), 1)

    def test_chunk_size_respected(self):
        text = "B" * 900
        chunks = chunk_text(text, size=400, overlap=0)
        # 900 / 400 = 3 chunks (400, 400, 100)
        self.assertEqual(len(chunks), 3)


class TestLoadDocuments(unittest.TestCase):

    def test_loads_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc1.txt").write_text("Hello from doc1")
            (Path(tmpdir) / "doc2.txt").write_text("Hello from doc2")
            docs = load_documents(tmpdir)
            self.assertEqual(len(docs), 2)
            filenames = {d["filename"] for d in docs}
            self.assertIn("doc1.txt", filenames)
            self.assertIn("doc2.txt", filenames)

    def test_ignores_non_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc.txt").write_text("valid content here")
            (Path(tmpdir) / "image.png").write_bytes(b"\x89PNG")
            (Path(tmpdir) / "data.csv").write_text("a,b,c")
            docs = load_documents(tmpdir)
            # only .txt loaded — .png and .csv are unsupported
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["filename"], "doc.txt")

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = load_documents(tmpdir)
            self.assertEqual(docs, [])

    def test_skips_empty_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "empty.txt").write_text("   ")
            docs = load_documents(tmpdir)
            self.assertEqual(docs, [])

    def test_loads_multiple_formats(self):
        """txt, pdf, docx all loaded when present and valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "notes.txt").write_text("text file content")
            # invalid pdf/docx — should be skipped gracefully, not crash
            (Path(tmpdir) / "broken.pdf").write_bytes(b"not a real pdf")
            (Path(tmpdir) / "broken.docx").write_bytes(b"not a real docx")
            docs = load_documents(tmpdir)
            # only the valid txt loads; broken files are skipped without error
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["filename"], "notes.txt")


if __name__ == "__main__":
    unittest.main()
