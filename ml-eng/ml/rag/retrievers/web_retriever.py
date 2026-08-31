"""
Supplemental web retrieval for RAG when internal corpora are weak.

Tier 1: Wikipedia (free, requests only).
Tier 2: Tavily news search (optional, requires TAVILY_API_KEY + langchain-tavily).

Guardrails:
- Per-UTC-day call counter for Tavily (``RAG_TAVILY_DAILY_LIMIT``, default 900)
  to stay under the free-tier ~1k/day cap. Counter is in-process only; a
  single replica that survives a day is sufficient for the freemium scale we
  ship at launch. Multi-replica enforcement is a post-launch task (use Redis).
- Rate-limit detection: when Tavily returns a 429 / quota error we stop
  immediately, do **not** retry, and surface a structured ``rate_limited``
  status so the graph can route to an "insufficient context" response
  instead of papering over with a stale internal document.
- Single short backoff (``RAG_TAVILY_BACKOFF_S``, default 2s) on transient
  network errors only — never on rate-limit responses.

Fail-soft: timeouts and unknown API errors return an ``error`` status with
empty items; the graph never sees a raised exception.
"""
from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import quote

import requests

from ml.rag.chatbot.generator import filter_context_items, is_usable_context_item
from ml.rag.chatbot.query_decomposer import resolve_retrieval_geographies, wants_africa_default_scope
from ml.rag.observability import get_observe_decorator, trace_elapsed_ms, update_current_span_metadata

logger = logging.getLogger(__name__)

_observe_span = get_observe_decorator()

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary"
_WIKI_UA = {"User-Agent": "OpenTrace-RAG/1.0 (agricultural advisory)"}
_OFFICIAL_WEB_DOMAINS = frozenset(
    {
        "faostat.org",
        "fao.org",
        "fews.net",
        "ipcinfo.org",
        "protectedplanet.net",
        "wdpa.org",
    }
)
_SUMMARY_THIN_CHARS = 120
_SECTION_EXTRACT_CAP = 1000
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "for",
        "to",
        "from",
        "with",
        "by",
        "as",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "have",
        "has",
        "had",
        "that",
        "this",
        "these",
        "those",
        "into",
        "about",
        "over",
        "under",
        "than",
        "then",
        "there",
        "their",
        "its",
        "it",
        "we",
        "you",
        "your",
        "our",
        "most",
        "least",
        "best",
        "worst",
        "please",
        "tell",
        "me",
        "show",
        "give",
        "explain",
    }
)

WebFallbackStatus = Literal["ok", "empty", "rate_limited", "disabled", "error"]


@dataclass
class WebFallbackResult:
    """Outcome of a supplemental web retrieval attempt.

    ``items`` holds usable web chunks (may be empty even when ``status == "ok"``
    if every result failed length / quality checks).

    ``status`` lets the graph distinguish "tried and got nothing usable" from
    "tried and got rate-limited / errored", which is what drives the
    "insufficient information" branch.
    """

    items: list[dict[str, Any]] = field(default_factory=list)
    status: WebFallbackStatus = "empty"
    reason: str = ""


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
    return _env_float("RAG_WEB_TIMEOUT_S", 5.0)


# --- Tavily daily quota counter (in-process, per UTC day) ---
_quota_lock = threading.Lock()
_quota_state: dict[str, Any] = {"date": "", "count": 0}


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _tavily_daily_limit() -> int:
    """Hard ceiling for Tavily calls per UTC day.

    Default 900 keeps us safely under the free tier's ~1000/day cap.
    Set ``RAG_TAVILY_DAILY_LIMIT`` to tune. Set to 0 to disable Tavily
    entirely (operational kill switch).
    """
    return _env_int("RAG_TAVILY_DAILY_LIMIT", 900)


def _tavily_backoff_s() -> float:
    return _env_float("RAG_TAVILY_BACKOFF_S", 2.0)


def _tavily_quota_available() -> bool:
    limit = _tavily_daily_limit()
    if limit <= 0:
        return False
    today = _utc_today()
    with _quota_lock:
        if _quota_state["date"] != today:
            _quota_state["date"] = today
            _quota_state["count"] = 0
        return int(_quota_state["count"]) < limit


def _tavily_record_call() -> None:
    today = _utc_today()
    with _quota_lock:
        if _quota_state["date"] != today:
            _quota_state["date"] = today
            _quota_state["count"] = 0
        _quota_state["count"] = int(_quota_state["count"]) + 1


def reset_tavily_quota() -> None:
    """Test helper: clear the in-process daily counter."""
    with _quota_lock:
        _quota_state["date"] = ""
        _quota_state["count"] = 0


def _context_kind(item: dict[str, Any]) -> str:
    return str(item.get("_context_kind") or item.get("source") or "").lower()


def _usable_reranked(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return filter_context_items(items or [])


def needs_web_fallback(
    reranked_context: list[dict[str, Any]],
    *,
    enabled: bool | None = None,
    task_mode: str | None = None,
    has_usable_bq: bool | None = None,
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

    mode = (task_mode or "").strip().lower()
    usable = _usable_reranked(reranked_context)
    min_chunks = _env_int("RAG_WEB_FALLBACK_MIN_CHUNKS", 3)

    if has_usable_bq is None:
        has_usable_bq = any(_context_kind(i) == "bigquery" for i in usable)
    if mode in ("fact_lookup", "data_export_only", "briefing") and has_usable_bq:
        return False
    if mode == "fact_lookup" and len(usable) >= min_chunks:
        has_news = any(_context_kind(i) in ("news", "news_article") for i in usable)
        if has_news or has_usable_bq:
            return False

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
    reranked = state.get("reranked_context") or []
    bq_results = state.get("bq_results") or []
    has_bq = any(
        _context_kind(r) == "bigquery" or str(r.get("source") or "").lower() == "bigquery"
        for r in bq_results
    )
    if needs_web_fallback(
        reranked,
        task_mode=str(state.get("task_mode") or ""),
        has_usable_bq=has_bq,
    ):
        return "web_fallback"
    return "generate"


def _dec_entities(decomposition: dict[str, Any] | None) -> list[str]:
    dec = decomposition or {}
    raw = dec.get("entities")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for ent in raw:
        s = str(ent).strip()
        if s and s not in out:
            out.append(s)
    return out


def _dec_domain_tokens(decomposition: dict[str, Any] | None) -> list[str]:
    dec = decomposition or {}
    raw = dec.get("domains")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for d in raw:
        for tok in _TOKEN_RE.findall(str(d).lower()):
            if tok not in out and tok not in _STOPWORDS:
                out.append(tok)
    return out


def _strip_question_tokens(query: str, *, max_tokens: int = 10) -> str:
    tokens = [
        t
        for t in _TOKEN_RE.findall(query or "")
        if t.lower() not in _STOPWORDS and len(t) > 1
    ]
    return " ".join(tokens[:max_tokens]).strip()


def _shape_wiki_queries(
    query: str,
    decomposition: dict[str, Any] | None,
    *,
    geo_override: str = "",
) -> list[str]:
    """Build 1–3 short Wikipedia search strings (entity+country first)."""
    dec = decomposition or {}
    countries = resolve_retrieval_geographies(
        geo_override=geo_override,
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    entities = _dec_entities(dec)
    country = countries[0] if countries else ""
    entity = entities[0] if entities else ""
    africa_default = bool(dec.get("africa_default")) or wants_africa_default_scope(query)

    shaped: list[str] = []

    def _add(s: str) -> None:
        text = " ".join(s.split()).strip()
        if len(text) < 2:
            return
        if text.lower() not in {x.lower() for x in shaped}:
            shaped.append(text[:120])

    if entity and country:
        _add(f"{entity} {country}")
    elif entity:
        _add(entity)
    elif country:
        _add(country)

    stripped = _strip_question_tokens(query)
    if stripped:
        if country and country.lower() not in stripped.lower():
            _add(f"{stripped} {country}")
        else:
            _add(stripped)

    if africa_default and not country:
        _add("agriculture Africa")
        if stripped and "africa" not in stripped.lower():
            _add(f"{stripped} Africa")

    if not shaped:
        # Last-resort short concat (legacy behavior, tighter cap).
        parts = [query.strip()]
        parts.extend(countries[:2])
        parts.extend(entities[:3])
        if africa_default:
            parts.append("Africa")
        _add(" ".join(p for p in parts if p)[:120])

    return shaped[:3]


def _build_wiki_search_query(
    query: str,
    decomposition: dict[str, Any] | None,
    *,
    geo_override: str = "",
) -> str:
    """Primary shaped Wikipedia/Tavily query (backward-compatible helper)."""
    shaped = _shape_wiki_queries(query, decomposition, geo_override=geo_override)
    return shaped[0] if shaped else ""


def _whole_word(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", haystack, flags=re.IGNORECASE) is not None


# Well-known non-African country tokens for soft wiki title filtering.
_NON_AFRICAN_TITLE_COUNTRIES: tuple[str, ...] = (
    "switzerland",
    "france",
    "germany",
    "china",
    "india",
    "brazil",
    "canada",
    "japan",
    "australia",
    "russia",
    "italy",
    "spain",
    "united kingdom",
    "united states",
    "usa",
    "uk",
    "netherlands",
    "belgium",
    "sweden",
    "norway",
    "poland",
    "mexico",
    "argentina",
    "chile",
    "south korea",
    "north korea",
)


def _wiki_title_passes(
    title: str,
    *,
    countries: list[str],
    entity_tokens: list[str],
    africa_default: bool = False,
) -> bool:
    """Soft geo/topic filter: drop titles that only name a different country."""
    t = (title or "").strip()
    if len(t) < 2:
        return False
    if africa_default and not countries:
        for foreign in _NON_AFRICAN_TITLE_COUNTRIES:
            if _whole_word(t, foreign):
                return False
        return True
    if not countries:
        return True
    allowed = {c.strip().lower() for c in countries if c.strip()}
    if any(_whole_word(t, c) for c in allowed):
        return True
    # Title names some other country-like token from a conflicting set:
    # if it contains "in <Country>" style and none of our countries, drop when
    # the foreign country appears as a whole word and no entity overlap.
    known_conflict = False
    for other in ("nigeria", "kenya", "ghana", "ethiopia", "uganda", "tanzania", "senegal", "rwanda"):
        if other in allowed:
            continue
        if _whole_word(t, other):
            known_conflict = True
            break
    if known_conflict:
        # Keep if topical entity still matches (e.g. "Maize" page while searching Kenya maize).
        if entity_tokens and any(_whole_word(t, e) for e in entity_tokens):
            return True
        return False
    return True


def _wiki_opensearch_titles(search_query: str, *, limit: int, timeout_s: float) -> list[str]:
    if not search_query.strip():
        return []
    params = {
        "action": "opensearch",
        "search": search_query,
        "limit": max(1, limit),
        "namespace": 0,
        "format": "json",
        "origin": "*",
    }
    try:
        resp = requests.get(_WIKI_API, params=params, timeout=timeout_s, headers=_WIKI_UA)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Wikipedia opensearch failed: %s", exc)
        return []

    # Response: [query, [titles...], [descriptions...], [urls...]]
    if not isinstance(data, list) or len(data) < 2:
        return []
    titles_raw = data[1] if isinstance(data[1], list) else []
    titles: list[str] = []
    for row in titles_raw:
        title = str(row or "").strip()
        if title and title not in titles:
            titles.append(title)
    return titles[:limit]


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
        resp = requests.get(_WIKI_API, params=params, timeout=timeout_s, headers=_WIKI_UA)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Wikipedia search failed: %s", exc)
        return []

    search = (data.get("query") or {}).get("search") or []
    titles: list[str] = []
    for row in search:
        title = str(row.get("title") or "").strip()
        if title and title not in titles:
            titles.append(title)
    return titles[:limit]


def _wiki_resolve_titles(search_query: str, *, limit: int, timeout_s: float) -> tuple[list[str], str]:
    """Prefer opensearch; fall back to list=search. Returns (titles, api_used)."""
    titles = _wiki_opensearch_titles(search_query, limit=limit, timeout_s=timeout_s)
    if titles:
        return titles, "opensearch"
    titles = _wiki_search_titles(search_query, limit=limit, timeout_s=timeout_s)
    return titles, ("search" if titles else "none")


def _wiki_page_summary(title: str, *, timeout_s: float) -> dict[str, Any] | None:
    safe_title = quote(title.replace(" ", "_"), safe="/()")
    url = f"{_WIKI_REST}/{safe_title}"
    try:
        resp = requests.get(
            url,
            timeout=timeout_s,
            headers=_WIKI_UA,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Wikipedia summary failed for %r: %s", title, exc)
        return None


def _html_to_text(raw_html: str) -> str:
    text = _HTML_TAG_RE.sub(" ", raw_html or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _wiki_section_extract(title: str, *, timeout_s: float) -> str | None:
    """Fetch first content section plain text when the lead summary is thin."""
    try:
        sections_resp = requests.get(
            _WIKI_API,
            params={
                "action": "parse",
                "page": title,
                "prop": "sections",
                "format": "json",
                "origin": "*",
            },
            timeout=timeout_s,
            headers=_WIKI_UA,
        )
        sections_resp.raise_for_status()
        sections = (sections_resp.json().get("parse") or {}).get("sections") or []
    except Exception as exc:
        logger.warning("Wikipedia sections failed for %r: %s", title, exc)
        return None

    section_idx: str | None = None
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        # Prefer first top-level content section (toclevel 1), skip empty indexes.
        if str(sec.get("toclevel") or "") == "1" and str(sec.get("index") or "").isdigit():
            section_idx = str(sec.get("index"))
            break
    if section_idx is None:
        for sec in sections:
            if isinstance(sec, dict) and str(sec.get("index") or "").isdigit():
                section_idx = str(sec.get("index"))
                break
    if section_idx is None:
        return None

    try:
        text_resp = requests.get(
            _WIKI_API,
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "section": section_idx,
                "format": "json",
                "origin": "*",
                "disableeditsection": "1",
            },
            timeout=timeout_s,
            headers=_WIKI_UA,
        )
        text_resp.raise_for_status()
        raw = ((text_resp.json().get("parse") or {}).get("text") or {}).get("*") or ""
    except Exception as exc:
        logger.warning("Wikipedia section text failed for %r: %s", title, exc)
        return None

    plain = _html_to_text(str(raw))
    if len(plain) < 40:
        return None
    return plain[:_SECTION_EXTRACT_CAP]


def _retrieve_wikipedia(
    search_query: str,
    *,
    top_k: int,
    timeout_s: float,
    countries: list[str] | None = None,
    entity_tokens: list[str] | None = None,
    africa_default: bool = False,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    titles, api_used = _wiki_resolve_titles(search_query, limit=max(top_k * 2, top_k), timeout_s=timeout_s)
    if stats is not None:
        stats["wiki_api"] = api_used
        stats["wiki_filtered_out"] = int(stats.get("wiki_filtered_out") or 0)
        stats["wiki_section_used"] = bool(stats.get("wiki_section_used"))

    country_list = [c for c in (countries or []) if str(c).strip()]
    entities = [e for e in (entity_tokens or []) if str(e).strip()]
    items: list[dict[str, Any]] = []
    for title in titles:
        if not _wiki_title_passes(
            title,
            countries=country_list,
            entity_tokens=entities,
            africa_default=africa_default,
        ):
            if stats is not None:
                stats["wiki_filtered_out"] = int(stats.get("wiki_filtered_out") or 0) + 1
            continue
        summary = _wiki_page_summary(title, timeout_s=timeout_s)
        if not summary:
            continue
        extract = str(summary.get("extract") or summary.get("description") or "").strip()
        section_used = False
        if len(extract) < _SUMMARY_THIN_CHARS:
            section = _wiki_section_extract(title, timeout_s=timeout_s)
            if section:
                extract = section
                section_used = True
                if stats is not None:
                    stats["wiki_section_used"] = True
        if len(extract) < 40:
            continue
        # Soft geo check on body when countries set and title didn't mention them.
        if country_list and not any(_whole_word(title, c) for c in country_list):
            if any(
                _whole_word(title, other)
                for other in ("nigeria", "kenya", "ghana", "ethiopia", "uganda", "tanzania", "senegal", "rwanda")
                if other not in {c.lower() for c in country_list}
            ) and not any(_whole_word(extract, c) for c in country_list):
                if stats is not None:
                    stats["wiki_filtered_out"] = int(stats.get("wiki_filtered_out") or 0) + 1
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
                    "wiki_section_used": section_used,
                },
            }
        )
        if len(items) >= top_k:
            break
    return items


def _tavily_results_to_items(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
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


def _retrieve_tavily(
    search_query: str,
    *,
    top_k: int,
    time_start: str | None,
    time_end: str | None,
) -> tuple[list[dict[str, Any]], WebFallbackStatus, str]:
    """Tier-2 Tavily news search with quota + rate-limit guardrails.

    Returns ``(items, status, reason)``.  Status values:
    - ``ok``           : got at least one usable item
    - ``empty``        : called Tavily, got zero usable items (or no API key)
    - ``rate_limited`` : 429 / quota signal, OR daily quota exhausted locally
    - ``disabled``     : Tavily tools not importable / not configured
    - ``error``        : transient error after backoff retry
    """
    try:
        from ml.web_data_mining.agentic.tavily_tools import (
            TAVILY_RATE_LIMIT_PREFIX,
            is_tavily_configured,
            tavily_search_news,
        )
    except ImportError:
        logger.warning("Tavily tools unavailable (ml.web_data_mining not importable)")
        return [], "disabled", "tavily tools not importable"

    if not is_tavily_configured():
        return [], "disabled", "TAVILY_API_KEY not set"

    if not _tavily_quota_available():
        logger.warning(
            "Tavily daily quota exhausted (limit=%d, count=%d). Skipping web search.",
            _tavily_daily_limit(),
            int(_quota_state.get("count") or 0),
        )
        return [], "rate_limited", "local daily quota exhausted"

    started = time.monotonic()
    _tavily_record_call()
    _text, results, err = tavily_search_news(
        search_query,
        max_results=top_k,
        start_date=time_start[:10] if time_start else None,
        end_date=time_end[:10] if time_end else None,
    )
    latency_ms = int((time.monotonic() - started) * 1000)

    if err:
        if err.startswith(TAVILY_RATE_LIMIT_PREFIX):
            logger.warning(
                "Tavily rate-limited (no retry, no fallback to stale doc). latency_ms=%d err=%s",
                latency_ms,
                err,
            )
            return [], "rate_limited", err
        # Single short backoff for transient errors; do NOT retry on rate-limit.
        backoff = _tavily_backoff_s()
        if backoff > 0:
            logger.info(
                "Tavily transient error (will retry once after %.1fs): %s", backoff, err
            )
            time.sleep(backoff)
            _tavily_record_call()
            started = time.monotonic()
            _text, results, err = tavily_search_news(
                search_query,
                max_results=top_k,
                start_date=time_start[:10] if time_start else None,
                end_date=time_end[:10] if time_end else None,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            if err:
                if err.startswith(TAVILY_RATE_LIMIT_PREFIX):
                    logger.warning("Tavily rate-limited on retry: %s", err)
                    return [], "rate_limited", err
                logger.warning(
                    "Tavily search failed after retry. latency_ms=%d err=%s",
                    latency_ms,
                    err,
                )
                return [], "error", err
        else:
            logger.warning(
                "Tavily search failed (no backoff configured). latency_ms=%d err=%s",
                latency_ms,
                err,
            )
            return [], "error", err

    items = _tavily_results_to_items(results, top_k)
    logger.info(
        "Tavily search ok: query_len=%d results=%d usable=%d latency_ms=%d",
        len(search_query),
        len(results or []),
        len(items),
        latency_ms,
    )
    return items, ("ok" if items else "empty"), ""


def _finalize_web_trace(
    result: WebFallbackResult,
    *,
    start: float,
    extra: dict[str, Any] | None = None,
) -> WebFallbackResult:
    source = result.reason if result.status == "ok" else result.status
    meta: dict[str, Any] = {
        "source": source,
        "status": result.status,
        "result_count": len(result.items),
        "latency_ms": trace_elapsed_ms(start),
    }
    if extra:
        meta.update(extra)
    update_current_span_metadata(meta)
    return result


def _url_domain(url: str) -> str:
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_official_web_url(url: str) -> bool:
    domain = _url_domain(url)
    if not domain:
        return False
    return domain in _OFFICIAL_WEB_DOMAINS or any(
        domain.endswith("." + d) or d in domain for d in _OFFICIAL_WEB_DOMAINS
    )


def _external_web_label(url: str) -> str:
    domain = _url_domain(url)
    if "faostat" in domain or domain.endswith("fao.org"):
        return "external (FAOSTAT/FAO)"
    if "protectedplanet" in domain or "wdpa" in domain:
        return "external (WDPA/Protected Planet)"
    if "fews" in domain:
        return "external (FEWS NET)"
    if "ipcinfo" in domain:
        return "external (IPC)"
    return "external"


def _label_external_web_item(item: dict[str, Any]) -> dict[str, Any]:
    meta = dict(item.get("metadata") or {})
    url = str(meta.get("url") or meta.get("source_url") or "")
    label = _external_web_label(url)
    meta["external_label"] = label
    meta["official_source"] = _is_official_web_url(url)
    content = str(item.get("content") or "").strip()
    if content and not content.startswith("["):
        content = f"[{label}] {content}"
    return {**item, "content": content, "metadata": meta}


def _typed_web_allowlist_only(decomposition: dict[str, Any] | None, query: str) -> bool:
    blob = (query or "").lower()
    if isinstance(decomposition, dict):
        blob += " " + " ".join(str(e) for e in (decomposition.get("entities") or []))
        blob += " " + str(decomposition.get("intent") or "")
    cues = (
        "protected area",
        "wdpa",
        "food balance",
        "import dependency",
        "ipc phase",
        "lean season",
        "fews",
        "outlook",
    )
    return any(c in blob for c in cues)


def _wiki_entity_tokens(decomposition: dict[str, Any] | None) -> list[str]:
    tokens: list[str] = []
    for ent in _dec_entities(decomposition):
        for tok in _TOKEN_RE.findall(ent):
            low = tok.lower()
            if low not in _STOPWORDS and low not in tokens:
                tokens.append(low)
    for tok in _dec_domain_tokens(decomposition):
        if tok not in tokens:
            tokens.append(tok)
    return tokens


@_observe_span(as_type="span", name="retrieval.web", capture_input=False, capture_output=False)
def retrieve_web_fallback_detailed(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    geo_override: str = "",
    time_start: str | None = None,
    time_end: str | None = None,
) -> WebFallbackResult:
    """
    Fetch supplemental web chunks with structured status reporting.

    Tries Wikipedia first (free). If Wikipedia returns nothing usable, escalates
    to Tavily — gated by the per-day quota and respecting rate-limit signals.

    The returned ``WebFallbackResult.status`` is what the graph uses to decide
    whether to fall through to the standard generator (``ok``) or to route to
    the "insufficient context" branch (``rate_limited``, ``error``, ``empty``,
    or ``disabled``).
    """
    t0 = time.perf_counter()
    if not web_fallback_enabled():
        return _finalize_web_trace(
            WebFallbackResult(status="disabled", reason="RAG_WEB_FALLBACK_ENABLED is off"),
            start=t0,
        )

    dec = decomposition or {}
    wiki_queries = _shape_wiki_queries(query, dec, geo_override=geo_override)
    primary_query = wiki_queries[0] if wiki_queries else ""
    if len(primary_query) < 2:
        return _finalize_web_trace(
            WebFallbackResult(status="empty", reason="search query too short"),
            start=t0,
        )

    countries = resolve_retrieval_geographies(
        geo_override=geo_override,
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    entity_tokens = _wiki_entity_tokens(dec)
    africa_default = bool(dec.get("africa_default")) or wants_africa_default_scope(query)

    timeout_s = _web_timeout_s()
    wiki_top_k = _env_int("RAG_WEB_WIKI_TOP_K", 2)
    tavily_top_k = _env_int("RAG_WEB_TAVILY_TOP_K", 2)
    total_cap = _env_int("RAG_WEB_TOP_K", 3)
    official_only = _typed_web_allowlist_only(dec, query)

    wiki_stats: dict[str, Any] = {
        "wiki_queries": wiki_queries,
        "wiki_api": "none",
        "wiki_filtered_out": 0,
        "wiki_section_used": False,
    }
    wiki_items: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    if not official_only:
        for wq in wiki_queries:
            need = wiki_top_k - len(wiki_items)
            if need <= 0:
                break
            q_stats: dict[str, Any] = {
                "wiki_filtered_out": 0,
                "wiki_section_used": False,
            }
            batch = _retrieve_wikipedia(
                wq,
                top_k=need,
                timeout_s=timeout_s,
                countries=countries,
                entity_tokens=entity_tokens,
                africa_default=africa_default,
                stats=q_stats,
            )
            if q_stats.get("wiki_api") and q_stats["wiki_api"] != "none":
                wiki_stats["wiki_api"] = q_stats["wiki_api"]
            wiki_stats["wiki_filtered_out"] = int(wiki_stats["wiki_filtered_out"]) + int(
                q_stats.get("wiki_filtered_out") or 0
            )
            if q_stats.get("wiki_section_used"):
                wiki_stats["wiki_section_used"] = True
            for item in batch:
                title_key = str((item.get("metadata") or {}).get("title") or "").strip().lower()
                if title_key and title_key in seen_titles:
                    continue
                if title_key:
                    seen_titles.add(title_key)
                wiki_items.append(item)
            if len(wiki_items) >= wiki_top_k:
                break

    if wiki_items:
        return _finalize_web_trace(
            WebFallbackResult(items=wiki_items[:total_cap], status="ok", reason="wikipedia"),
            start=t0,
            extra=wiki_stats,
        )

    tavily_items, status, reason = _retrieve_tavily(
        primary_query,
        top_k=tavily_top_k,
        time_start=time_start,
        time_end=time_end,
    )
    if official_only:
        tavily_items = [
            _label_external_web_item(it)
            for it in tavily_items
            if _is_official_web_url(str((it.get("metadata") or {}).get("url") or ""))
        ]
    else:
        tavily_items = [_label_external_web_item(it) for it in tavily_items]
    if status == "ok" and tavily_items:
        return _finalize_web_trace(
            WebFallbackResult(items=tavily_items[:total_cap], status="ok", reason="tavily"),
            start=t0,
            extra=wiki_stats,
        )
    return _finalize_web_trace(
        WebFallbackResult(items=[], status=status, reason=reason or "no usable web results"),
        start=t0,
        extra=wiki_stats,
    )


def retrieve_web_fallback(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    geo_override: str = "",
    time_start: str | None = None,
    time_end: str | None = None,
) -> list[dict[str, Any]]:
    """
    Backwards-compatible wrapper: returns only the items list.

    New callers should prefer :func:`retrieve_web_fallback_detailed` so they
    can inspect ``status`` and route to an "insufficient context" response
    instead of running generation on stale internal context.
    """
    return retrieve_web_fallback_detailed(
        query,
        decomposition,
        geo_override=geo_override,
        time_start=time_start,
        time_end=time_end,
    ).items


def format_web_chunk_for_context(item: dict[str, Any]) -> dict[str, Any]:
    """Prefix web chunk content for the generator context block."""
    kind = _context_kind(item)
    raw_meta = item.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
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
