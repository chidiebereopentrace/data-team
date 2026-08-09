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
    "answer_lang",
    "acf_band",
    "acf_band_label",
    "acf_score",
    "acf_explanation",
    "acf_claim_level",
    "acf_question_type",
    "artifacts",
    "langfuse_trace_id",
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
    (
        "Swahili lang",
        "Habari, nipe taarifa za kilimo Kenya.",
        {
            "plan_type": "Farmers",
            "category": "Farmers",
            "user_profile": {
                "country": "Kenya",
                "plan_type": "Farmers",
                "category": "Farmers",
            },
        },
    ),
    (
        "Pidgin lang",
        "Wetin be the maize price for Abuja?",
        {},
    ),
    (
        "Agri export CSV",
        "Export maize production data for Nigeria as a CSV",
        {
            "plan_type": "Agribusinesses",
            "category": "Agribusinesses",
            "user_profile": {
                "country": "Nigeria",
                "plan_type": "Agribusinesses",
                "category": "Agribusinesses",
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
    if result.get("is_greeting_query"):
        return "greeting"
    if result.get("is_out_of_scope_query"):
        return "out_of_scope"
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
    if route in ("greeting", "out_of_scope"):
        return "decompose → generate_social → END"
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
    """Map POST /v1/chat/{plan} JSON (ChatSuccessResponse) into inspector shape.

    Production response fields: assistant_message, citations, acf (nested ACFSignal),
    session_id, usage, request_id, created_at, plan_type, langfuse_trace_id, artifacts.
    There is no 'trace' field in production — retrieval counts are unavailable in HTTP mode.
    """
    usage_raw = payload.get("usage")
    usage: dict[str, int] = {}
    if isinstance(usage_raw, dict):
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens") or 0),
            "output_tokens": int(usage_raw.get("output_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }

    # ACF is a nested ACFSignal object in production; flatten to the keys
    # render_metrics_row already reads (acf_band, acf_score, acf_explanation, …).
    acf = _as_dict(payload.get("acf"))
    # plan_type echoed back by the API; fall back to kwargs when absent.
    plan_type = payload.get("plan_type") or kwargs.get("plan_type")
    user_profile = kwargs.get("user_profile")

    return {
        "answer": payload.get("assistant_message") or "",
        "citations": payload.get("citations") or [],
        "error": payload.get("error"),
        "session_id": payload.get("session_id"),
        "usage": usage,
        "latency_ms": latency_ms,
        "plan_type": plan_type,
        "category": _as_dict(user_profile).get("category"),
        "user_profile": user_profile,
        "langfuse_trace_id": payload.get("langfuse_trace_id"),
        "artifacts": payload.get("artifacts") or [],
        # ACF flattened — mirrors the flat keys consumed by render_metrics_row.
        "acf_band": acf.get("band") or "",
        "acf_band_label": acf.get("band_label") or "",
        "acf_score": acf.get("score"),
        "acf_explanation": acf.get("explanation") or acf.get("note") or "",
        "acf_claim_level": acf.get("claim_level"),
        "acf_question_type": acf.get("question_type"),
        # HTTP mode: no decomposition or retrieval counts from this endpoint.
        "decomposition": {},
        "bq_table_candidates": [],
        "vector_news_results": [],
        "vector_academic_results": [],
        "bq_results": [],
        "vector_ota_results": [],
        "merged_context": [],
        "reranked_context": [],
        "web_results": [],
        "_backend_mode": "http_api",
        "_query": query,
    }


def query_via_http_api(
    base_url: str,
    query: str,
    *,
    kwargs: dict[str, Any],
    session_id: str | None = None,
    trace_id: str | None = None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """POST /v1/chat/{plan} and return a normalized inspector result dict.

    Targets the plan-scoped production routes (ML-034) which return ChatSuccessResponse.
    Plan slug is derived from kwargs['plan_type']; defaults to 'integrated' when unset.
    ChatRequest has extra='forbid' — only message, session_id, user_profile are sent.
    """
    import time

    # Derive plan slug from kwargs['plan_type']; lowercase maps all plan IDs to slugs.
    plan_type = str(kwargs.get("plan_type") or "").strip()
    plan_slug = plan_type.lower() if plan_type else "integrated"
    url = base_url.rstrip("/") + f"/v1/chat/{plan_slug}"

    # ChatRequest fields only — no include_trace, no internal top_k params.
    body: dict[str, Any] = {"message": query}
    if session_id:
        body["session_id"] = session_id
    profile = kwargs.get("user_profile")
    if isinstance(profile, dict) and profile.get("plan_type") and profile.get("category"):
        body["user_profile"] = profile

    t0 = time.perf_counter()
    headers: dict[str, str] = {}
    if trace_id:
        headers["X-Langfuse-Trace-Id"] = trace_id
    resp = requests.post(url, json=body, timeout=timeout_s, headers=headers or None)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected /v1/chat response type: {type(payload)!r}")
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
    trace_id = str(result.get("langfuse_trace_id") or "").strip()
    if trace_id:
        st.caption(f"Langfuse trace: `{trace_id}`")


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

    # ACF Path B + language + per-plan model (ML-029 / ML-039 / ML-041)
    acf_band = str(result.get("acf_band_label") or result.get("acf_band") or "").strip()
    acf_score = result.get("acf_score")
    answer_lang = str(result.get("answer_lang") or "").strip()
    if acf_band or acf_score is not None or answer_lang:
        from ml.rag.chatbot.plan_policy import model_for_plan
        from ml.rag.llm_chat import llm_model_id
        plan_type = str(result.get("plan_type") or "").strip()
        model_used = model_for_plan(plan_type) or llm_model_id()
        model_short = model_used.split("/")[-1] if "/" in model_used else model_used

        r3 = st.columns(4)
        with r3[0]:
            st.metric("ACF band", acf_band or "—")
        with r3[1]:
            score_disp = f"{acf_score:.0f}/100" if isinstance(acf_score, (int, float)) else "—"
            st.metric("ACF score", score_disp)
        with r3[2]:
            st.metric("answer_lang", answer_lang or "—")
        with r3[3]:
            st.metric("LLM model", model_short or "—")

        expl = str(result.get("acf_explanation") or result.get("acf_note") or "").strip()
        if expl:
            st.caption(f"ACF explanation: {expl[:200]}")


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


def render_artifacts(result: dict[str, Any]) -> None:
    """Render export artifacts (CSV / chart / DOCX / PDF) with download links.

    Artifacts are only present on Agribusinesses and Integrated plan responses
    (ML-030). Each ArtifactItem has: id, kind, filename, mime_type, url,
    summary, citation_ids, byte_size.

    URL handling (Option A):
      - https:// URLs  → st.link_button (clickable download)
      - file://  URLs  → st.code (dev-only; can't open cross-origin in browser)
    """
    artifacts = _as_list(result.get("artifacts"))
    real = [a for a in artifacts if isinstance(a, dict)]
    if not real:
        return

    st.subheader(f"Export artifacts ({len(real)})")
    for art in real:
        kind = str(art.get("kind") or "file").upper()
        fname = str(art.get("filename") or "export")
        url = str(art.get("url") or "").strip()
        summary = str(art.get("summary") or "")
        byte_size = art.get("byte_size")
        size_str = f"{byte_size:,} bytes" if isinstance(byte_size, int) else ""

        header = f"**[{kind}]** {fname}" + (f"  ·  {size_str}" if size_str else "")
        with st.expander(header, expanded=True):
            if summary:
                st.caption(summary)
            if url.startswith("https://"):
                st.link_button(f"Download {fname}", url)
            elif url.startswith("file://"):
                st.caption("Local dev artifact — copy path to access:")
                st.code(url)
            else:
                st.caption(f"URL: {url or '(none)'}")


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

        render_artifacts(result)

        if backend_mode == "http_api" and result.get("_http_trace"):
            st.subheader("HTTP trace (counts)")
            st.json(result.get("_http_trace"))

        render_raw_json(result)
