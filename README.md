# RAG with Endee Vector Database

A production-ready **Retrieval Augmented Generation (RAG)** system built on [Endee](https://github.com/endee-io/endee) — a high-performance open-source C++ vector database designed for AI search workloads.

---

## Project Overview & Problem Statement

Traditional search is keyword-based — it matches exact words, not meaning. When you ask a question about your own documents, a keyword search fails to understand context, synonyms, or intent.

**RAG solves this** by combining semantic vector search with document retrieval:

1. Documents are chunked, embedded into vectors, and stored in Endee
2. At query time, the question is embedded and the most semantically similar chunks are retrieved
3. Those chunks are returned as a grounded, cited answer — no hallucination, no LLM required

This project implements a complete RAG pipeline using **Endee as the sole vector database** for all storage and retrieval. It works fully offline — no OpenAI key, no external API, no internet connection needed.

---

## System Design & Technical Approach

```
INGESTION PIPELINE (run once to load knowledge base)
─────────────────────────────────────────────────────
  Documents (.txt / .pdf / .docx)
       │
       ▼
  chunk_text()          400-char overlapping chunks (80-char overlap)
       │
       ▼
  Embedder              all-MiniLM-L6-v2 → 384-dim float32 vectors (local, no API)
       │
       ▼
  EndeeClient           HTTP POST → Endee stores vector + metadata + filter fields
       │
       ▼
  Endee (HNSW index)    persists vectors on disk via MDBX embedded key-value store


QUERY PIPELINE (runs on every user question)
─────────────────────────────────────────────
  User question
       │
       ▼
  Embedder.embed_one()  same model → 384-dim query vector
       │
       ▼
  EndeeClient.search()  ANN search via HNSW → top-k nearest chunks (msgpack response)
       │
       ▼
  Retriever             unpacks results → [{text, filename, chunk_index, score}]
       │
       ▼
  Generator             combines top 3 chunks → cited answer (no LLM needed)
       │
       ▼
  {answer, sources}     returned with filename + score citations
```

### Component Responsibilities

| File | Responsibility |
|---|---|
| `endee_client.py` | HTTP wrapper for Endee REST API (create, insert, search, delete) |
| `embedder.py` | Local sentence-transformer embedding — no API key needed |
| `ingest.py` | Load documents → chunk → embed → upsert into Endee |
| `retriever.py` | Embed query → search Endee → return ranked chunks |
| `generator.py` | Combine top retrieved chunks into a cited answer |
| `rag_pipeline.py` | Orchestrates retriever + generator; CLI entry point |
| `tests/` | 50 unit tests across all components, fully offline |

### Key Technical Decisions

- **Chunking**: 400-char character-level chunks with 80-char overlap — overlap prevents context loss at boundaries
- **Embedding model**: `all-MiniLM-L6-v2` — 384 dimensions, fast, runs locally, no API key
- **Vector precision**: `float32` — full precision cosine similarity
- **No-LLM mode**: answers are built directly from the top 3 retrieved chunks with source citations — works with zero API keys
- **Metadata storage**: each vector stores `{text, filename, chunk_index}` as JSON in Endee's `meta` field
- **Filter support**: `filename` stored as a filterable field for per-document scoped queries

---

## How Endee Is Used

Endee is the core of this project — every piece of knowledge flows through it.

### 1. Index Creation

```python
client.create_index(
    index_name="rag_docs",
    dim=384,              # matches all-MiniLM-L6-v2 output dimension
    space_type="cosine",  # cosine similarity for normalized embeddings
    precision="float32",
    M=16,                 # HNSW graph connections per node
    ef_con=128,           # construction-time search depth
)
```

### 2. Vector Insert with Metadata

Endee's insert API expects a top-level JSON array. `meta` and `filter` are serialized as JSON strings:

```python
client.insert_vectors("rag_docs", [{
    "id": "550e8400-e29b-41d4-a716",
    "vector": [0.12, -0.04, 0.87, ...],     # 384 floats
    "metadata": {
        "text": "HNSW stands for Hierarchical Navigable Small World...",
        "filename": "hnsw_algorithm.txt",
        "chunk_index": 0,
    },
    "filter": {"filename": "hnsw_algorithm.txt"},
}])
```

### 3. Similarity Search

```python
results = client.search(
    index_name="rag_docs",
    vector=query_embedding,   # 384-dim float list
    k=5,                      # top-k results
    ef=128,                   # search-time exploration factor
    filters=[{"filename": {"$eq": "hnsw_algorithm.txt"}}],  # optional
)
# returns: [{"id": "...", "score": 0.93, "metadata": {...}}, ...]
```

Endee returns results as **msgpack** binary. The response is a list of tuples:
`[similarity, id, meta_bytes, filter_str, norm, vector]`

The client decodes this automatically and returns clean Python dicts.

### 4. Why Endee for RAG

- **HNSW algorithm** — sub-linear ANN search, scales to millions of vectors
- **Metadata + filter fields** — store chunk text alongside vectors, filter by filename
- **Embedded storage (MDBX)** — no separate database process, data persists on disk
- **Docker deployment** — single container, zero configuration

---

## Project Structure

```
rag-endee/
├── docs/                        # knowledge base — drop files here
│   ├── endee_overview.txt
│   ├── hnsw_algorithm.txt
│   ├── sparse_search.txt
│   ├── rag_concepts.txt
│   ├── data_analytics.txt
│   ├── machine_learning.txt
│   ├── python_programming.txt
│   └── web_development.txt
├── tests/
│   ├── test_endee_client.py     # 13 tests — HTTP client
│   ├── test_embedder.py         #  4 tests — embedding
│   ├── test_ingest.py           # 11 tests — chunking + document loading
│   ├── test_retriever.py        #  6 tests — retrieval logic
│   ├── test_generator.py        #  8 tests — prompt + answer generation
│   └── test_rag_pipeline.py     #  6 tests — end-to-end orchestration
├── endee_client.py
├── embedder.py
├── ingest.py
├── retriever.py
├── generator.py
├── rag_pipeline.py
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Setup & Execution

### Prerequisites

- Python 3.11+
- Docker Desktop

### Step 1 — Start Endee

```bash
docker-compose up -d
```

Verify it's healthy:
```bash
curl http://localhost:8080/api/v1/health
# {"status":"ok","timestamp":...}
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure

```bash
cp .env.example .env
```

`.env` settings (all defaults work out of the box — no API key needed):
```
ENDEE_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=          # leave empty if auth disabled

# LLM options (all optional — pick one or leave both empty)
OLLAMA_MODEL=              # e.g. llama3.2 — free local LLM via Ollama
OLLAMA_URL=http://localhost:11434
OPENAI_API_KEY=            # paid OpenAI key

INDEX_NAME=rag_docs
EMBED_MODEL=all-MiniLM-L6-v2
TOP_K=5
```

Answer generation priority:
1. **Ollama** (free, local) — if `OLLAMA_MODEL` is set
2. **OpenAI** — if `OPENAI_API_KEY` is set
3. **No-LLM fallback** — always works, returns top 3 doc chunks with citations

### Step 4 — Ingest documents

```bash
python ingest.py --docs ./docs --index rag_docs
```

```
[Ingest] Loaded 8 documents from ./docs
[Ingest] Total chunks to embed: 61
[Ingest] Inserted 61/61 chunks
[Ingest] Done. 61 chunks stored in index 'rag_docs'.
```

### Step 5 — Query

```bash
# Single question
python rag_pipeline.py --query "What is HNSW?"

# Filter to one document
python rag_pipeline.py --query "What is time series analysis?" --file data_analytics.txt

# Interactive mode
python rag_pipeline.py --interactive

# Upload your own file then ask questions about it
python rag_pipeline.py --upload
```

Example output:
```
============================================================
Question: What is HNSW?
============================================================

Answer:
Based on your documents:

[1] From hnsw_algorithm.txt:
HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for
approximate nearest neighbor search...

[2] From endee_overview.txt:
Endee uses HNSW as its core indexing algorithm...

Sources:
  [1] hnsw_algorithm.txt (chunk 0, score=0.6878)
  [2] hnsw_algorithm.txt (chunk 1, score=0.5840)
  [3] endee_overview.txt (chunk 2, score=0.4164)
```

### Step 6 — Run tests

```bash
python -m pytest tests/ -v
# 55 passed — all offline, no server or API key needed
```

---

## Optional: Ollama (Free Local LLM)

If you want real LLM-generated answers instead of raw doc excerpts, install [Ollama](https://ollama.com):

```bash
# 1. Download and install Ollama from https://ollama.com
# 2. Pull a model (llama3.2 is small and fast)
ollama pull llama3.2

# 3. Set in your .env
OLLAMA_MODEL=llama3.2
```

That's it. The pipeline will automatically route through Ollama when `OLLAMA_MODEL` is set. If Ollama isn't running, it gracefully falls back to no-LLM mode.

---

## Agile Development Approach

Built feature-by-feature — each sprint adds one component with tests before moving to the next:

| Sprint | Feature | Tests |
|---|---|---|
| 1 | `EndeeClient` — health, create, insert, search, delete | 13 |
| 2 | `Embedder` — local sentence-transformer, no API key | 4 |
| 3 | `ingest.py` — chunking, .txt / .pdf / .docx loading | 11 |
| 4 | `Retriever` — embed query, search Endee, ranked results | 6 |
| 5 | `Generator` — no-LLM fallback + Ollama + OpenAI routing | 13 |
| 6 | `RAGPipeline` — orchestrator + CLI (query / interactive / upload) | 6 |
| 7 | Knowledge base — 8 domain docs ingested into Endee | manual |
| **Total** | | **55** |

Each sprint was verified working before the next started. All tests mock external dependencies (Endee server, embedding model) so the full suite runs offline in under 2 seconds.

---

## CLI Reference

| Command | Description |
|---|---|
| `python ingest.py --docs ./docs --index rag_docs` | Ingest all docs into Endee |
| `python rag_pipeline.py --query "..."` | Single question |
| `python rag_pipeline.py --query "..." --file doc.txt` | Question scoped to one file |
| `python rag_pipeline.py --interactive` | Interactive Q&A session |
| `python rag_pipeline.py --upload` | Upload a file, ingest it, then Q&A |
| `python -m pytest tests/ -v` | Run all 50 tests |

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `ENDEE_URL` | `http://localhost:8080` | Endee server URL |
| `ENDEE_AUTH_TOKEN` | _(empty)_ | Auth token (leave empty if disabled) |
| `OLLAMA_MODEL` | _(empty)_ | Ollama model name e.g. `llama3.2` — free local LLM |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | _(empty)_ | Optional — leave empty for no-LLM mode |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model (only used if key is set) |
| `INDEX_NAME` | `rag_docs` | Endee index name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `TOP_K` | `5` | Chunks retrieved per query |

---

## Supported File Types

| Format | Parser |
|---|---|
| `.txt` | built-in |
| `.pdf` | PyMuPDF (`fitz`) |
| `.docx` | python-docx |
