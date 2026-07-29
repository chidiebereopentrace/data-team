"""OpenRouter / Cohere batch rerank client for the RAG stack."""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

from ml.rag.llm_chat import llm_uses_openrouter
from ml.rag.observability import get_openrouter_run_id, openrouter_sessions_enabled

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL_ID = "cohere/rerank-4-pro"
_SOFT_FAIL_HTTP = frozenset({401, 402, 403, 410, 429, 502, 503})


def _llm_api_key() -> str:
    return os.environ.get("RAG_LLM_API_KEY", "").strip()


def _openrouter_extra_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_APP_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def rerank_enabled() -> bool:
    return os.environ.get("RAG_LLM_RERANK", "off").strip().lower() not in {"off", "0", "false"}


def rerank_mode() -> str:
    """
    Rerank strategy when ``rerank_enabled()`` is true.

    - ``openrouter``: batch POST /rerank (default when OpenRouter base URL is set)
    - ``llm``: legacy one chat completion per chunk
    - ``boost``: source-boost sort only
    """
    raw = os.environ.get("RAG_RERANK_MODE", "").strip().lower()
    if raw in {"openrouter", "llm", "boost"}:
        return raw
    if llm_uses_openrouter():
        return "openrouter"
    return "boost"


def rerank_model_id() -> str:
    return (
        os.environ.get("RAG_RERANK_MODEL_ID", DEFAULT_RERANK_MODEL_ID).strip()
        or DEFAULT_RERANK_MODEL_ID
    )


def rerank_timeout_s() -> float:
    return float(os.environ.get("RAG_RERANK_TIMEOUT_S", "30") or 30)


def rerank_max_doc_chars() -> int:
    return max(100, int(os.environ.get("RAG_RERANK_MAX_DOC_CHARS", "2000") or 2000))


def openrouter_rerank_url() -> str | None:
    base = os.environ.get("RAG_LLM_BASE_URL", "").strip().rstrip("/")
    if not base or not llm_uses_openrouter():
        return None
    return f"{base}/rerank"


def openrouter_rerank_configured() -> bool:
    return bool(openrouter_rerank_url() and _llm_api_key())


def _parse_rerank_results(data: dict[str, Any]) -> list[tuple[int, float]]:
    """Parse OpenRouter/Cohere-style rerank response into (index, score) pairs."""
    raw_results = data.get("results")
    if raw_results is None and isinstance(data.get("data"), dict):
        raw_results = data["data"].get("results")
    if not isinstance(raw_results, list):
        return []

    out: list[tuple[int, float]] = []
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        if idx is None:
            idx = row.get("document_index")
        score = row.get("relevance_score")
        if score is None:
            score = row.get("score")
        if idx is None or score is None:
            continue
        try:
            out.append((int(idx), float(score)))
        except (TypeError, ValueError):
            continue
    return out


def openrouter_rerank(
    query: str,
    documents: list[str],
    *,
    top_n: int,
    model: str | None = None,
    timeout_s: float | None = None,
) -> list[tuple[int, float]]:
    """
    Batch-rerank documents via OpenRouter ``POST /rerank``.

    Returns ``(document_index, relevance_score)`` pairs, or ``[]`` on failure.
    """
    url = openrouter_rerank_url()
    if not url or not documents:
        return []

    api_key = _llm_api_key()
    if not api_key:
        logger.warning("openrouter_rerank: no API key (set RAG_LLM_API_KEY)")
        return []

    max_chars = rerank_max_doc_chars()
    docs = [str(d or "")[:max_chars] for d in documents]
    effective_timeout = rerank_timeout_s() if timeout_s is None else timeout_s
    effective_top_n = max(1, min(int(top_n), len(docs)))

    headers: dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    headers.update(_openrouter_extra_headers())
    if openrouter_sessions_enabled():
        run_id = get_openrouter_run_id()
        if run_id:
            headers["x-session-id"] = run_id[:256]

    payload = {
        "model": model or rerank_model_id(),
        "query": query,
        "documents": docs,
        "top_n": effective_top_n,
    }
    if openrouter_sessions_enabled():
        run_id = get_openrouter_run_id()
        if run_id:
            payload["session_id"] = run_id[:256]

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=effective_timeout)
        if resp.status_code in _SOFT_FAIL_HTTP:
            logger.warning(
                "openrouter_rerank: HTTP %s from %s (model=%s)",
                resp.status_code,
                url,
                payload["model"],
            )
            return []
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            logger.warning("openrouter_rerank: non-object JSON from %s", url)
            return []
        results = _parse_rerank_results(data)
        if not results:
            logger.warning("openrouter_rerank: empty results from %s", url)
        return results
    except requests.Timeout:
        logger.warning(
            "openrouter_rerank: timed out after %.0fs (model=%s, url=%s)",
            effective_timeout,
            payload["model"],
            url,
        )
        return []
    except Exception:
        logger.exception(
            "openrouter_rerank: request failed (model=%s, url=%s)",
            payload["model"],
            url,
        )
        return []
