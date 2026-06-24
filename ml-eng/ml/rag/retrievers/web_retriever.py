"""
Supplemental web retrieval for RAG when internal corpora are weak.

Tier 1: Wikipedia (free, requests only).
Tier 2: Tavily news search (optional, requires TAVILY_API_KEY + langchain-tavily).

Fail-soft: timeouts and API errors return empty lists; never raise through the graph.
"""
from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import quote

import requests

from ml.rag.chatbot.generator import filter_context_items, is_usable_context_item
from ml.rag.chatbot.query_decomposer import resolve_retrieval_geographies

logger = logging.getLogger(__name__)

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary"


def _env_on(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "on", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def web_fallback_enabled() -> bool:
    return _env_on("RAG_WEB_FALLBACK_ENABLED", default=False)


def _web_timeout_s() -> float:
    return _env_float("RAG_WEB_TIMEOUT_S", 8.0)


def _context_kind(item: dict[str, Any]) -> str:
    return str(item.get("_context_kind") or item.get("source") or "").lower()


def _usable_reranked(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return filter_context_items(items or [])


def needs_web_fallback(
    reranked_context: list[dict[str, Any]],
    *,
    enabled: bool | None = None,
) -> bool:
    """
    Return True when supplemental web retrieval should run.

    Triggers when enabled and either:
    - usable chunk count < RAG_WEB_FALLBACK_MIN_CHUNKS (default 3), or
    - no usable news and no usable BQ rows (only academic/OTA/other left).
    """
    if enabled is None:
        enabled = web_fallback_enabled()
    if not enabled:
        return False

    usable = _usable_reranked(reranked_context)
    min_chunks = _env_int("RAG_WEB_FALLBACK_MIN_CHUNKS", 3)
    if len(usable) < min_chunks:
        return True

    has_news = any(_context_kind(i) in ("news", "news_article") for i in usable)
    has_bq = any(_context_kind(i) == "bigquery" for i in usable)
    if not has_news and not has_bq:
        return True

    llm_rerank_on = os.environ.get("RAG_LLM_RERANK", "off").lower() not in {"off", "0", "false"}
    min_score = _env_float("RAG_WEB_FALLBACK_MIN_RERANK_SCORE", -1.0)
    if llm_rerank_on and min_score >= 0 and usable:
        top_score = max(
            float(i.get("_rerank_score", i.get("_llm_score", -1.0)) or -1.0) for i in usable
        )
        if top_score < min_score:
            return True

    return False


def route_after_rerank(state: dict[str, Any]) -> str:
    """Graph routing: 'web_fallback' or 'generate'."""
    if needs_web_fallback(state.get("reranked_context") or []):
        return "web_fallback"
    return "generate"


def _build_wiki_search_query(
    query: str,
    decomposition: dict[str, Any] | None,
    *,
    geo_override: str = "",
) -> str:
    parts: list[str] = [query.strip()]
    dec = decomposition or {}
    countries = resolve_retrieval_geographies(
        geo_override=geo_override,
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    parts.extend(countries[:2])
    entities = dec.get("entities") if isinstance(dec.get("entities"), list) else []
    for ent in entities[:3]:
        s = str(ent).strip()
        if s:
            parts.append(s)
    text = " ".join(p for p in parts if p).strip()
    return text[:300] if len(text) > 300 else text


def _wiki_search_titles(search_query: str, *, limit: int, timeout_s: float) -> list[str]:
    if not search_query.strip():
        return []
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_query,
        "srlimit": limit,
        "format": "json",
        "origin": "*",
    }
    try:
        resp = requests.get(_WIKI_API, params=params, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Wikipedia search failed: %s", exc)
        return []

    search = (data.get("query") or {}).get("search") or []
    titles: list[str] = []
    for row in search:
        title = str(row.get("title") or "").strip()
        if title:
            titles.append(title)
    return titles[:limit]


def _wiki_page_summary(title: str, *, timeout_s: float) -> dict[str, Any] | None:
    safe_title = quote(title.replace(" ", "_"), safe="/()")
    url = f"{_WIKI_REST}/{safe_title}"
    try:
        resp = requests.get(
            url,
            timeout=timeout_s,
            headers={"User-Agent": "OpenTrace-RAG/1.0 (agricultural advisory)"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Wikipedia summary failed for %r: %s", title, exc)
        return None


def _retrieve_wikipedia(
    search_query: str,
    *,
    top_k: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    titles = _wiki_search_titles(search_query, limit=top_k, timeout_s=timeout_s)
    items: list[dict[str, Any]] = []
    for title in titles:
        summary = _wiki_page_summary(title, timeout_s=timeout_s)
        if not summary:
            continue
        extract = str(summary.get("extract") or summary.get("description") or "").strip()
        if len(extract) < 40:
            continue
        page_url = str(
            summary.get("content_urls", {}).get("desktop", {}).get("page")
            or summary.get("canonicalurl")
            or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        )
        items.append(
            {
                "content": extract,
                "source": "web_wikipedia",
                "_context_kind": "web_wikipedia",
                "metadata": {
                    "title": title,
                    "url": page_url,
                    "provider": "wikipedia",
                },
            }
        )
    return items


def _retrieve_tavily(
    search_query: str,
    *,
    top_k: int,
    time_start: str | None,
    time_end: str | None,
) -> list[dict[str, Any]]:
    try:
        from ml.web_data_mining.agentic.tavily_tools import is_tavily_configured, tavily_search_news
    except ImportError:
        logger.warning("Tavily tools unavailable (ml.web_data_mining not importable)")
        return []

    if not is_tavily_configured():
        return []

    _text, results, err = tavily_search_news(
        search_query,
        max_results=top_k,
        start_date=time_start[:10] if time_start else None,
        end_date=time_end[:10] if time_end else None,
    )
    if err:
        logger.warning("Tavily search failed: %s", err)
        return []

    items: list[dict[str, Any]] = []
    for row in results[:top_k]:
        title = str(row.get("title") or "").strip()
        content = str(row.get("content") or "").strip()
        url = str(row.get("url") or "").strip()
        body = content or title
        if len(body) < 30:
            continue
        items.append(
            {
                "content": body,
                "source": "web_search",
                "_context_kind": "web_search",
                "metadata": {
                    "title": title or "Web result",
                    "url": url,
                    "provider": "tavily",
                },
            }
        )
    return items


def retrieve_web_fallback(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    geo_override: str = "",
    time_start: str | None = None,
    time_end: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch supplemental web chunks: Wikipedia first, then Tavily if Wikipedia is empty.
    """
    if not web_fallback_enabled():
        return []

    search_query = _build_wiki_search_query(
        query, decomposition, geo_override=geo_override
    )
    if len(search_query) < 5:
        return []

    timeout_s = _web_timeout_s()
    wiki_top_k = _env_int("RAG_WEB_WIKI_TOP_K", 2)
    tavily_top_k = _env_int("RAG_WEB_TAVILY_TOP_K", 2)
    total_cap = _env_int("RAG_WEB_TOP_K", 3)

    items = _retrieve_wikipedia(search_query, top_k=wiki_top_k, timeout_s=timeout_s)
    if not items:
        items = _retrieve_tavily(
            search_query,
            top_k=tavily_top_k,
            time_start=time_start,
            time_end=time_end,
        )

    return items[:total_cap]


def format_web_chunk_for_context(item: dict[str, Any]) -> dict[str, Any]:
    """Prefix web chunk content for the generator context block."""
    kind = _context_kind(item)
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    title = str(meta.get("title") or "").strip()
    body = str(item.get("content") or "").strip()
    if kind == "web_wikipedia":
        label = f"[Wikipedia | {title}]" if title else "[Wikipedia]"
    else:
        label = f"[Web | {title}]" if title else "[Web]"
    return {
        **item,
        "content": f"{label} {body}".strip(),
        "source": item.get("source", kind),
        "_context_kind": kind,
        "metadata": meta,
    }
