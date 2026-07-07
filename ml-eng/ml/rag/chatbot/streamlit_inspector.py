"""
Pipeline inspector helpers for the Streamlit RAG QA UI.

Pure functions for route inference and HTTP response normalization; Streamlit render
functions for full retrieval / flow observability after each query.
"""
from __future__ import annotations

import os
from typing import Any, Literal

import requests
import streamlit as st

BackendMode = Literal["in_process", "http_api"]

INSPECTOR_JSON_KEYS: tuple[str, ...] = (
    "is_meta_query",
    "is_product_query",
    "insufficient_context",
    "decomposition",
    "plan_type",
    "category",
    "user_profile",
    "geo_override",
    "bq_table_candidates",
    "vector_news_results",
    "vector_academic_results",
    "vector_ota_results",
    "bq_results",
    "bq_sql_queries",
    "merged_context",
    "reranked_context",
    "web_results",
    "web_fallback_status",
    "web_fallback_reason",
    "answer",
    "citations",
    "usage",
    "error",
    "latency_ms",
    "_backend_mode",
    "_http_trace",
)

PRESET_QUERIES: list[tuple[str, str, dict[str, Any]]] = [
    ("Identity", "Who are you?", {}),
    ("Product", "What is OpenTrace?", {}),
    ("RAG", "Maize yields in Kenya 2020", {}),
    (
        "Farmers + Ghana",
        "What are maize production trends?",
        {
            "plan_type": "Farmers",
            "category": "Farmers",
            "user_profile": {
                "country": "Ghana",
                "plan_type": "Farmers",
                "category": "Farmers",
            },
        },
    ),
]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def infer_pipeline_route(result: dict[str, Any]) -> str:
    """Derive the terminal graph path from a run_rag() result dict."""
    if result.get("is_meta_query"):
        return "meta"
    if result.get("is_product_query"):
        return "product"
    if result.get("insufficient_context"):
        return "insufficient"
    if result.get("web_results"):
        return "full_rag + web_fallback"
    return "full_rag"


def _flow_narrative(route: str) -> str:
    if route == "meta":
        return "decompose → generate_meta → END"
    if route == "product":
        return "decompose → generate_product → END"
    if route == "insufficient":
        return (
            "decompose → parallel_retrieve → bq_retrieve → merge → rerank "
            "→ web_fallback → insufficient_context → END"
        )
    if route == "full_rag + web_fallback":
        return (
            "decompose → parallel_retrieve → bq_retrieve → merge → rerank "
            "→ web_fallback → generate → END"
        )
    return (
        "decompose → parallel_retrieve → bq_retrieve → merge → rerank → generate → END"
    )


def show_sql_debug() -> bool:
    return os.environ.get("RAG_SHOW_SQL_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def debug_default_enabled() -> bool:
    raw = os.environ.get("RAG_STREAMLIT_DEBUG_DEFAULT", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def normalize_http_response(
    payload: dict[str, Any],
    *,
    latency_ms: float,
    query: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Map POST /query JSON into a shape the inspector can render (counts only)."""
    trace = _as_dict(payload.get("trace"))
    usage_raw = payload.get("usage")
    usage: dict[str, int] = {}
    if isinstance(usage_raw, dict):
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens") or 0),
            "output_tokens": int(usage_raw.get("output_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }

    plan_type = kwargs.get("plan_type")
    category = kwargs.get("category")
    user_profile = kwargs.get("user_profile")

    return {
        "answer": payload.get("answer") or "",
        "citations": payload.get("citations") or [],
        "error": payload.get("error"),
        "session_id": payload.get("session_id"),
        "usage": usage,
        "decomposition": trace.get("decomposition") or {},
        "bq_table_candidates": [None] * int(trace.get("bq_table_candidates_count") or 0),
        "vector_news_results": [None] * int(trace.get("vector_news_count") or 0),
        "vector_academic_results": [None] * int(trace.get("vector_academic_count") or 0),
        "bq_results": [],
        "vector_ota_results": [],
        "merged_context": [None] * int(trace.get("merged_context_count") or 0),
        "reranked_context": [None] * int(trace.get("reranked_context_count") or 0),
        "web_results": [],
        "latency_ms": latency_ms,
        "plan_type": plan_type,
        "category": category,
        "user_profile": user_profile,
        "_backend_mode": "http_api",
        "_http_trace": trace,
        "_query": query,
    }


def query_via_http_api(
    base_url: str,
    query: str,
    *,
    kwargs: dict[str, Any],
    session_id: str | None = None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """POST /query and return a normalized inspector result dict."""
    import time

    url = base_url.rstrip("/") + "/query"
    body: dict[str, Any] = {
        "query": query,
        "include_trace": True,
    }
    if session_id:
        body["session_id"] = session_id
    profile = kwargs.get("user_profile")
    if isinstance(profile, dict) and profile.get("plan_type") and profile.get("category"):
        body["user_profile"] = profile
    for key in (
        "time_start_override",
        "time_end_override",
        "news_top_k",
        "academic_top_k",
        "bq_top_k",
        "rerank_top_k",
        "ota_top_k",
    ):
        if key in kwargs and kwargs[key] is not None:
            body[key] = kwargs[key]

    t0 = time.perf_counter()
    resp = requests.post(url, json=body, timeout=timeout_s)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected /query response type: {type(payload)!r}")
    result = normalize_http_response(payload, latency_ms=latency_ms, query=query, kwargs=kwargs)
    return result


def render_chunk_rows(items: list[dict[str, Any]], *, preview_chars: int = 600) -> None:
    """Render retrieval/rerank chunks as collapsible rows."""
    real_items = [x for x in items if isinstance(x, dict)]
    if not real_items:
        if items and all(x is None for x in items):
            st.info("Count-only (HTTP API mode — switch to In-process for chunk detail).")
        else:
            st.info("No items.")
        return
    for i, it in enumerate(real_items, start=1):
        content = str(it.get("content") or "")
        score = it.get("score")
        rerank_score = it.get("_rerank_score")
        llm_score = it.get("_llm_score")
        raw_meta = it.get("metadata")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        source = it.get("source") or it.get("_context_kind") or "?"
        title_bits = [f"#{i}", f"[{source}]"]
        if isinstance(score, (int, float)):
            title_bits.append(f"score={score:.4f}")
        if isinstance(rerank_score, (int, float)):
            title_bits.append(f"rerank={rerank_score:.4f}")
        if isinstance(llm_score, (int, float)):
            title_bits.append(f"llm={llm_score:.4f}")
        for k in (
            "section_title",
            "label",
            "source_file",
            "title",
            "url",
            "doi",
            "table_name",
            "geo_country_primary",
            "country",
            "published_at",
        ):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                title_bits.append(f"{k}={v.strip()[:50]}")
                break
        header = " · ".join(title_bits)
        with st.expander(header, expanded=False):
            score_parts: list[str] = []
            if isinstance(score, (int, float)):
                score_parts.append(f"retrieval score: {score:.6f}")
            if isinstance(rerank_score, (int, float)):
                score_parts.append(f"rerank score: {rerank_score:.6f}")
            if isinstance(llm_score, (int, float)):
                score_parts.append(f"llm score: {llm_score:.6f}")
            if score_parts:
                st.caption(" · ".join(score_parts) + f"  ·  source: {source}")
            if meta:
                st.json(
                    {k: v for k, v in meta.items() if v is not None and v != ""},
                    expanded=False,
                )
            preview = content if len(content) <= preview_chars else content[:preview_chars] + "…"
            st.markdown(preview if preview else "_(empty content)_")


def render_flow_strip(route: str, result: dict[str, Any]) -> None:
    st.markdown(f"**Route:** `{route}`")
    st.caption(_flow_narrative(route))
    if result.get("_backend_mode") == "http_api":
        st.warning(
            "HTTP API mode: chunk lists are counts only. Use **In-process** backend for full retrieval detail."
        )


def render_request_context(
    result: dict[str, Any],
    *,
    query: str = "",
    memory_summary_len: int = 0,
    memory_recent_count: int = 0,
) -> None:
    dec = _as_dict(result.get("decomposition"))
    geo = _as_list(dec.get("geography"))
    profile = _as_dict(result.get("user_profile"))

    cols = st.columns(4)
    with cols[0]:
        st.metric("plan_type", str(result.get("plan_type") or profile.get("plan_type") or "—"))
    with cols[1]:
        st.metric("category", str(result.get("category") or profile.get("category") or "—"))
    with cols[2]:
        st.metric("profile country", str(profile.get("country") or "—"))
    with cols[3]:
        st.metric("geo override", str(result.get("geo_override") or "—"))

    if query:
        st.caption(f"Query: {query}")
    if geo:
        st.caption(f"Decomposition geography: {', '.join(str(g) for g in geo)}")
    st.caption(
        f"Memory: summary {memory_summary_len} chars · recent turns {memory_recent_count} messages"
    )


def render_metrics_row(result: dict[str, Any], *, latency_ms: float | None = None) -> None:
    usage = _as_dict(result.get("usage"))
    lat = latency_ms if latency_ms is not None else result.get("latency_ms")

    r1 = st.columns(7)
    with r1[0]:
        st.metric("BQ table matches", len(result.get("bq_table_candidates") or []))
    with r1[1]:
        st.metric("BQ rows", len(result.get("bq_results") or []))
    with r1[2]:
        st.metric("News", len(result.get("vector_news_results") or []))
    with r1[3]:
        st.metric("Research", len(result.get("vector_academic_results") or []))
    with r1[4]:
        st.metric("OTA", len(result.get("vector_ota_results") or []))
    with r1[5]:
        st.metric("Web", len(result.get("web_results") or []))
    with r1[6]:
        st.metric("→ generator", len(result.get("reranked_context") or []))

    r2 = st.columns(4)
    with r2[0]:
        st.metric("Latency (ms)", f"{lat:.0f}" if isinstance(lat, (int, float)) else "—")
    with r2[1]:
        st.metric("Input tokens", int(usage.get("input_tokens") or 0))
    with r2[2]:
        st.metric("Output tokens", int(usage.get("output_tokens") or 0))
    with r2[3]:
        st.metric("Total tokens", int(usage.get("total_tokens") or 0))

    status = result.get("web_fallback_status")
    if status:
        reason = result.get("web_fallback_reason") or ""
        st.caption(f"Web fallback: **{status}**" + (f" — {reason}" if reason else ""))


def render_sql_panel(result: dict[str, Any]) -> None:
    if not show_sql_debug():
        return
    bq_sql_list = list(result.get("bq_sql_queries") or [])
    if not bq_sql_list:
        bq_rows = result.get("bq_results") or []
        seen_sql: set[str] = set()
        for row in bq_rows:
            if not isinstance(row, dict):
                continue
            s = str((row.get("metadata") or {}).get("sql") or "").strip()
            if s and s not in seen_sql:
                seen_sql.add(s)
                bq_sql_list.append(s)
    if not bq_sql_list:
        return
    with st.expander(f"Generated SQL ({len(bq_sql_list)})", expanded=False):
        st.caption("Internal only. Set RAG_SHOW_SQL_DEBUG=0 to hide.")
        for i, sql in enumerate(bq_sql_list, start=1):
            st.caption(f"Query {i}")
            st.code(sql, language="sql")


def render_retrieval_tabs(result: dict[str, Any]) -> None:
    tabs = st.tabs([
        f"News ({len(result.get('vector_news_results') or [])})",
        f"Research ({len(result.get('vector_academic_results') or [])})",
        f"OTA ({len(result.get('vector_ota_results') or [])})",
        f"BQ descriptions ({len(result.get('bq_table_candidates') or [])})",
        f"BQ rows ({len(result.get('bq_results') or [])})",
        f"Merged ({len(result.get('merged_context') or [])})",
        f"Generator input ({len(result.get('reranked_context') or [])})",
        f"Web ({len(result.get('web_results') or [])})",
    ])
    with tabs[0]:
        render_chunk_rows(list(result.get("vector_news_results") or []))
    with tabs[1]:
        render_chunk_rows(list(result.get("vector_academic_results") or []))
    with tabs[2]:
        render_chunk_rows(list(result.get("vector_ota_results") or []))
    with tabs[3]:
        render_chunk_rows(list(result.get("bq_table_candidates") or []))
    with tabs[4]:
        render_chunk_rows(list(result.get("bq_results") or []))
    with tabs[5]:
        render_chunk_rows(list(result.get("merged_context") or []))
    with tabs[6]:
        st.caption("Exact context block passed to the generator, in order.")
        render_chunk_rows(list(result.get("reranked_context") or []))
    with tabs[7]:
        status = result.get("web_fallback_status")
        if status:
            st.caption(f"Status: {status} · {result.get('web_fallback_reason') or ''}")
        render_chunk_rows(list(result.get("web_results") or []))


def render_raw_json(result: dict[str, Any]) -> None:
    slim: dict[str, Any] = {}
    for key in INSPECTOR_JSON_KEYS:
        if key in result:
            val = result[key]
            if isinstance(val, list) and val and val[0] is None:
                slim[key] = f"<{len(val)} items — HTTP count only>"
            else:
                slim[key] = val
    with st.expander("Raw inspector JSON", expanded=False):
        st.json(slim)


def render_pipeline_inspector(
    result: dict[str, Any],
    *,
    latency_ms: float | None = None,
    backend_mode: BackendMode = "in_process",
    query: str = "",
    memory_summary_len: int = 0,
    memory_recent_count: int = 0,
) -> None:
    """Full pipeline inspector panel for the last query."""
    route = infer_pipeline_route(result)
    with st.expander("Pipeline inspector (last run)", expanded=True):
        render_flow_strip(route, result)
        render_request_context(
            result,
            query=query or str(result.get("_query") or ""),
            memory_summary_len=memory_summary_len,
            memory_recent_count=memory_recent_count,
        )

        dec = result.get("decomposition") or {}
        st.subheader("Query decomposition")
        st.json(dec if isinstance(dec, dict) else {})

        st.subheader("Retrieval metrics")
        render_metrics_row(result, latency_ms=latency_ms)

        render_sql_panel(result)

        st.subheader("Retrieved data by source")
        render_retrieval_tabs(result)

        if backend_mode == "http_api" and result.get("_http_trace"):
            st.subheader("HTTP trace (counts)")
            st.json(result.get("_http_trace"))

        render_raw_json(result)
