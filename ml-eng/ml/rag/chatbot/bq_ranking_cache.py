"""Session-scoped reuse of structured BQ ranking results for follow-up queries."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from ml.rag.chatbot.export_intent import detect_export_intent
from ml.rag.chatbot.generator import is_ranking_numeric_query

_FOLLOW_UP_RE = re.compile(
    r"\b("
    r"top\s+\d+|"
    r"list\s+(?:the\s+)?(?:countries|ranking|results)|"
    r"show\s+(?:me\s+)?(?:the\s+)?(?:list|ranking|top|results)|"
    r"who\s+(?:was|is)\s+(?:second|third|#?\d+|ranked\s+#?\d+)"
    r")\b",
    re.IGNORECASE,
)


def _s(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def fingerprint_ranking(
    decomposition: dict[str, Any] | None,
    sql: str | None,
    template: str | None,
) -> str:
    """Stable cache key from time window, SQL, and template."""
    dec = decomposition if isinstance(decomposition, dict) else {}
    parts = [
        _s(template),
        _s(sql),
        _s(dec.get("time_start")),
        _s(dec.get("time_end")),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def cache_entry_from_bq_results(
    bq_results: list[dict[str, Any]],
    *,
    query: str,
    decomposition: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract the first ranked_table item into a serializable cache entry."""
    dec = decomposition if isinstance(decomposition, dict) else {}
    for item in bq_results or []:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if meta.get("bq_enrichment") != "ranked_table":
            continue
        ranked_rows = meta.get("ranked_rows")
        if not isinstance(ranked_rows, list) or not ranked_rows:
            continue
        sql = _s(meta.get("sql"))
        template = _s(meta.get("template"))
        return {
            "fingerprint": fingerprint_ranking(dec, sql, template),
            "query": query,
            "decomposition": {
                k: dec.get(k)
                for k in ("time_start", "time_end", "geography", "entities", "domains")
                if dec.get(k) is not None
            },
            "item": {
                "content": item.get("content"),
                "source": item.get("source", "bigquery"),
                "metadata": meta,
            },
        }
    return None


def bq_results_from_cache(entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Rehydrate BQ context items from a cache entry."""
    if not isinstance(entry, dict):
        return []
    item = entry.get("item")
    if not isinstance(item, dict):
        return []
    return [dict(item)]


def _same_time_window(
    decomposition: dict[str, Any] | None,
    cached_dec: dict[str, Any] | None,
) -> bool:
    dec = decomposition if isinstance(decomposition, dict) else {}
    cached = cached_dec if isinstance(cached_dec, dict) else {}
    ts = _s(dec.get("time_start"))
    te = _s(dec.get("time_end"))
    cts = _s(cached.get("time_start"))
    cte = _s(cached.get("time_end"))
    if ts or te or cts or cte:
        return ts == cts and te == cte
    return True


def is_ranking_follow_up(
    query: str,
    decomposition: dict[str, Any] | None,
    cached: dict[str, Any] | None,
) -> bool:
    """True when a follow-up can reuse cached ranked_table results."""
    if not isinstance(cached, dict):
        return False
    item = cached.get("item")
    if not isinstance(item, dict):
        return False
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    fp = fingerprint_ranking(decomposition, _s(meta.get("sql")), _s(meta.get("template")))
    if fp != _s(cached.get("fingerprint")):
        return False
    if not _same_time_window(decomposition, cached.get("decomposition")):
        return False

    q = query or ""
    if detect_export_intent(q):
        return True
    if _FOLLOW_UP_RE.search(q):
        return True
    if is_ranking_numeric_query(q) and len(q.split()) <= 12:
        return True
    return False


__all__ = [
    "bq_results_from_cache",
    "cache_entry_from_bq_results",
    "fingerprint_ranking",
    "is_ranking_follow_up",
]
