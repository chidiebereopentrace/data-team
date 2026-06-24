"""
Reranker node: takes merged retrieval results and reranks them (e.g. cross-encoder or LLM)
before passing to the generator.

Uses configured LLM backend (HF router or RAG_LLM_BASE_URL). If unavailable, pass-through top_k.
"""
from __future__ import annotations

import os
from typing import Any

from ml.rag.llm_chat import llm_chat_complete, llm_model_id


def _llm_configured() -> bool:
    return bool(os.environ.get("HF_API_TOKEN") or os.environ.get("RAG_LLM_BASE_URL", "").strip())


def _score_with_llama(query: str, text: str) -> float:
    """
    Ask Llama to score how relevant `text` is to `query` on [0, 1].
    This is a simple, low-throughput reranker: one call per chunk.
    """
    prompt = (
        "You are a ranking model. Given a user question and a context chunk, "
        "return a single floating point number between 0 and 1 indicating how relevant "
        "the context is to answering the question. Respond with ONLY the number.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{text}\n\n"
        "Relevance score (0-1):"
    )
    raw = llm_chat_complete(
        [{"role": "user", "content": prompt}],
        model=llm_model_id(),
        max_tokens=8,
        temperature=0.0,
        timeout_s=30,
    )
    if not raw:
        return -1.0
    try:
        token = raw.split()[0]
        return float(token)
    except Exception:
        return -1.0


def rerank(
    query: str,
    context_items: list[dict[str, Any]],
    top_k: int = 5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Rerank context_items by relevance to query. Each item should have "content" (or "text").

    - If an LLM backend is configured and RAG_LLM_RERANK != \"off\", scores each chunk.
    - Otherwise, returns first top_k items (original order).
    """
    if not context_items:
        return []

    use_llm = os.environ.get("RAG_LLM_RERANK", "off").lower() not in {"off", "0", "false"}
    content_key = "content" if any("content" in c for c in context_items) else "text"

    _SOURCE_BOOST = {
        "bigquery": 0.12,
        "news": 0.04,
        "academic": 0.06,
        "web_wikipedia": 0.0,
        "web_search": 0.0,
    }

    if not use_llm or not _llm_configured():
        scored = []
        for i, item in enumerate(context_items):
            text = item.get(content_key) or item.get("text", str(item))
            kind = item.get("_context_kind") or item.get("source") or ""
            boost = _SOURCE_BOOST.get(str(kind).lower(), 0.0)
            scored.append(
                {
                    **item,
                    "content": text,
                    "_order": i,
                    "_llm_score": -1.0,
                    "_source_boost": boost,
                    "_rerank_score": boost,
                }
            )
        scored.sort(key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
        return scored[:top_k]

    scored = []
    for i, item in enumerate(context_items):
        text = (item.get(content_key) or item.get("text", str(item)))[:2000]
        score = _score_with_llama(query, text)
        kind = item.get("_context_kind") or item.get("source") or ""
        boost = _SOURCE_BOOST.get(str(kind).lower(), 0.0)
        adjusted = score + boost if score >= 0 else score
        scored.append(
            {
                **item,
                "content": text,
                "_order": i,
                "_llm_score": score,
                "_source_boost": boost,
                "_rerank_score": adjusted,
            }
        )

    scored.sort(key=lambda x: x.get("_rerank_score", x.get("_llm_score", -1.0)), reverse=True)
    return scored[:top_k]
