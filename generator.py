"""
Generator: takes retrieved context chunks and a question,
builds a prompt, and produces an answer.

Priority order:
  1. Ollama (local LLM) — if OLLAMA_MODEL is set in .env
  2. OpenAI             — if OPENAI_API_KEY is set in .env
  3. No-LLM fallback    — always works, returns top 3 chunks from docs
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['filename']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in context_chunks
    )
    return (
        "You are a helpful assistant. Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say 'I don't have enough information to answer that.'\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generate an answer given a question and retrieved context chunks.

    Routing:
      - OLLAMA_MODEL set → use Ollama (local LLM, free)
      - OPENAI_API_KEY set → use OpenAI
      - neither → no-LLM fallback (returns top 3 doc chunks)
    """
    if not context_chunks:
        return "No relevant context found in the knowledge base."

    if OLLAMA_MODEL:
        return _call_ollama(question, context_chunks)

    if OPENAI_API_KEY:
        return _call_openai(_build_prompt(question, context_chunks))

    return _local_fallback(question, context_chunks)


def _call_ollama(question: str, chunks: list[dict]) -> str:
    """Call a locally running Ollama model via its REST API."""
    prompt = _build_prompt(question, chunks)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip()
        return f"[Ollama error] HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        # Ollama not running — fall back to doc chunks
        return _local_fallback(question, chunks, note="(Ollama not running — showing doc excerpts)")
    except Exception as e:
        return f"[Ollama error] {e}"


def _call_openai(prompt: str) -> str:
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error] {e}"


def _local_fallback(question: str, chunks: list[dict], note: str = "") -> str:
    """
    No-LLM mode: combines top retrieved chunks into a readable cited answer.
    Works with zero API keys — purely from document content.
    """
    header = f"Based on your documents:{f'  {note}' if note else ''}\n"
    lines = [header]
    for i, chunk in enumerate(chunks[:3], 1):
        lines.append(f"[{i}] From {chunk['filename']}:\n{chunk['text'].strip()}\n")
    return "\n".join(lines)
