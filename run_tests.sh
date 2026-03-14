#!/usr/bin/env bash
# Run all unit tests for the RAG pipeline
set -e

cd "$(dirname "$0")"

echo "=== Running RAG-Endee Unit Tests ==="
python -m pytest tests/ -v --tb=short 2>&1
echo "=== All tests passed ==="
