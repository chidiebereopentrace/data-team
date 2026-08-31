"""
Pipeline inspector helpers for the Streamlit RAG QA UI.

Pure functions for route inference and HTTP response normalization; Streamlit render
functions for full retrieval / flow observability after each query.
"""
from __future__ import annotations

import os
import re
from typing import Any, Literal

import requests
import streamlit as st

from ml.rag.chatbot.generator import usable_context_after_geo_purity

BackendMode = Literal["in_process", "http_api"]

_INSPECTOR_META_SKIP = frozenset({"tier", "as_of_date"})
_RERANK_PREVIEW_PREFIX_RE = re.compile(r"^\[[^\]]+\]\n?", re.MULTILINE)


def _strip_rerank_preview_prefix(content: str) -> str:
    """Remove a leading rerank metadata line like ``[geo=Ghana; year=2020]``."""
    return _RERANK_PREVIEW_PREFIX_RE.sub("", content, count=1).lstrip()

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
    "bq_sql_plan",
    "vector_news_results",
    "vector_academic_papers_results",
    "vector_policies_results",
    "vector_public_reports_results",
    "vector_formation_results",
    "vector_academic_results",
    "vector_ota_results",
    "bq_results",
    "bq_sql_queries",
    "bq_sql_debug",
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
    "export_intent",
    "task_mode",
    "analytical_mode",
    "generation_plan",
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
    if result.get("is_help_query"):
        return "help"
    if result.get("is_product_query"):
        return "product"
    if result.get("is_greeting_query"):
        return "greeting"
    if result.get("is_out_of_scope_query"):
        return "out_of_scope"
    if result.get("is_language_unknown"):
        return "language_unknown"
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
    if route == "help":
        return "decompose → generate_product → END"
    if route in ("greeting", "out_of_scope"):
        return "decompose → generate_social → END"
    if route == "language_unknown":
        return "decompose → generate_language_help → END"
    if route == "insufficient":
        return (
            "decompose → parallel_retrieve → bq_reason → bq_retrieve → merge → rerank "
            "→ web_fallback → insufficient_context → END"
        )
    if route == "full_rag + web_fallback":
        return (
            "decompose → parallel_retrieve → bq_reason → bq_retrieve → merge → rerank "
            "→ web_fallback → generate → END"
        )
    return (
        "decompose → parallel_retrieve → bq_reason → bq_retrieve → merge → rerank → generate → END"
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


def normalize_query_response(
    payload: dict[str, Any],
    *,
    latency_ms: float,
    query: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Map POST /query/{plan} JSON (QueryResponse) into inspector shape.

    When include_trace is true, trace carries decomposition, retrieval counts,
    and BQ SQL debug fields for the pipeline inspector.
    """
    trace = _as_dict(payload.get("trace"))
    usage_raw = payload.get("usage")
    usage: dict[str, int] = {}
    if isinstance(usage_raw, dict):
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens") or 0),
            "output_tokens": int(usage_raw.get("output_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }

    acf = _as_dict(payload.get("acf"))
    plan_type = kwargs.get("plan_type")
    user_profile = kwargs.get("user_profile")
    bq_sql_plan = trace.get("bq_sql_plan") if trace else None
    if not isinstance(bq_sql_plan, dict):
        bq_sql_plan = {}

    return {
        "answer": payload.get("answer") or "",
        "citations": payload.get("citations") or [],
        "error": payload.get("error"),
        "session_id": payload.get("session_id"),
        "usage": usage,
        "latency_ms": latency_ms,
        "plan_type": plan_type,
        "category": _as_dict(user_profile).get("category"),
        "user_profile": user_profile,
        "langfuse_trace_id": payload.get("langfuse_trace_id") or trace.get("langfuse_trace_id"),
        "artifacts": payload.get("artifacts") or [],
        "acf_band": acf.get("band") or "",
        "acf_band_label": acf.get("band_label") or "",
        "acf_score": acf.get("score"),
        "acf_explanation": acf.get("explanation") or acf.get("note") or "",
        "acf_claim_level": acf.get("claim_level"),
        "acf_question_type": acf.get("question_type"),
        "decomposition": trace.get("decomposition") if trace else {},
        "bq_sql_plan": bq_sql_plan,
        "bq_sql_queries": list(trace.get("bq_sql_queries") or []) if trace else [],
        "bq_sql_debug": list(trace.get("bq_sql_debug") or []) if trace else [],
        "sql_source": trace.get("sql_source") if trace else None,
        "bq_cache_hit": trace.get("bq_cache_hit") if trace else None,
        "bq_nl2sql_ms": trace.get("bq_nl2sql_ms") if trace else None,
        "bq_execute_ms": trace.get("bq_execute_ms") if trace else None,
        "bq_table_candidates": [],
        "vector_news_results": [],
        "vector_academic_papers_results": [],
        "vector_policies_results": [],
        "vector_public_reports_results": [],
        "vector_formation_results": [],
        "vector_academic_results": [],
        "bq_results": [],
        "vector_ota_results": [],
        "merged_context": [],
        "reranked_context": [],
        "web_results": [],
        "_backend_mode": "http_api",
        "_http_trace": trace,
        "_query": query,
    }


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
        "vector_academic_papers_results": [],
        "vector_policies_results": [],
        "vector_public_reports_results": [],
        "vector_formation_results": [],
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
    """POST /query/{plan} with include_trace and return a normalized inspector result dict.

    Uses the RAG API debug trace (decomposition, retrieval counts, BQ SQL) for the
    pipeline inspector. Plan slug is derived from kwargs['plan_type']; defaults to
    'integrated' when unset.
    """
    import time

    plan_type = str(kwargs.get("plan_type") or "").strip()
    plan_slug = plan_type.lower() if plan_type else "integrated"
    url = base_url.rstrip("/") + f"/query/{plan_slug}"

    body: dict[str, Any] = {"query": query, "include_trace": True}
    if session_id:
        body["session_id"] = session_id
    profile = kwargs.get("user_profile")
    if isinstance(profile, dict) and profile:
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
        val = kwargs.get(key)
        if val is not None:
            if isinstance(val, str):
                val = val.strip()
            if val != "":
                body[key] = val

    t0 = time.perf_counter()
    headers: dict[str, str] = {}
    if trace_id:
        headers["X-Langfuse-Trace-Id"] = trace_id
    resp = requests.post(url, json=body, timeout=timeout_s, headers=headers or None)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected /query response type: {type(payload)!r}")
    return normalize_query_response(payload, latency_ms=latency_ms, query=query, kwargs=kwargs)


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
                    {
                        k: v
                        for k, v in meta.items()
                        if k not in _INSPECTOR_META_SKIP and v is not None and v != ""
                    },
                    expanded=False,
                )
            preview_raw = _strip_rerank_preview_prefix(content)
            preview = (
                preview_raw
                if len(preview_raw) <= preview_chars
                else preview_raw[:preview_chars] + "…"
            )
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

    if result.get("task_mode"):
        plan = _as_dict(result.get("bq_sql_plan"))
        intents = _as_list(plan.get("query_intents"))
        mode = str(result.get("task_mode"))
        if mode == "analytical" or result.get("analytical_mode"):
            st.info(
                f"**task_mode=analytical** — BQ intents: {len(intents)} · "
                f"skip_bq={plan.get('skip_bq')} · rationale={plan.get('rationale') or '—'}"
            )
            expanded = _as_list(dec.get("expanded_regions"))
            if expanded:
                st.caption(f"Expanded regions: {', '.join(str(x) for x in expanded)}")
        else:
            st.caption(
                f"task_mode={mode}"
                + (f" · BQ intents={len(intents)}" if intents else "")
                + (f" · rationale={plan.get('rationale')}" if plan.get("rationale") else "")
            )
    elif result.get("analytical_mode"):
        plan = _as_dict(result.get("bq_sql_plan"))
        intents = _as_list(plan.get("query_intents"))
        st.info(
            f"**Analytical mode** — BQ intents: {len(intents)} · "
            f"skip_bq={plan.get('skip_bq')} · rationale={plan.get('rationale') or '—'}"
        )
        expanded = _as_list(dec.get("expanded_regions"))
        if expanded:
            st.caption(f"Expanded regions: {', '.join(str(x) for x in expanded)}")

    gen_plan = _as_dict(result.get("generation_plan"))
    if gen_plan:
        shape = str(gen_plan.get("answer_shape") or "—")
        priority = gen_plan.get("evidence_priority") or []
        priority_head = priority[0] if isinstance(priority, list) and priority else "—"
        st.caption(
            f"generation_plan: shape={shape} · evidence_priority={priority_head}"
            + (f" · rationale={gen_plan.get('rationale')}" if gen_plan.get("rationale") else "")
        )

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

    r1 = st.columns(8)
    with r1[0]:
        st.metric("BQ table matches", len(result.get("bq_table_candidates") or []))
    with r1[1]:
        st.metric("BQ rows", len(result.get("bq_results") or []))
    with r1[2]:
        st.metric("News", len(result.get("vector_news_results") or []))
    with r1[3]:
        st.metric("Academic", len(result.get("vector_academic_papers_results") or []))
    with r1[4]:
        st.metric("Policy", len(result.get("vector_policies_results") or []))
    with r1[5]:
        st.metric("Public", len(result.get("vector_public_reports_results") or []))
    with r1[6]:
        st.metric("Formation", len(result.get("vector_formation_results") or []))
    with r1[7]:
        st.metric("OTA", len(result.get("vector_ota_results") or []))

    r1b = st.columns(3)
    with r1b[0]:
        st.metric("Web", len(result.get("web_results") or []))
    with r1b[1]:
        st.metric("→ reranked", len(result.get("reranked_context") or []))
    with r1b[2]:
        st.metric("→ cited", len(result.get("citations") or []))

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


def _bq_was_attempted(result: dict[str, Any]) -> bool:
    """True when the BQ path ran (candidates, plan, SQL debug, or results)."""
    if result.get("bq_sql_debug") or result.get("bq_sql_queries") or result.get("bq_results"):
        return True
    if result.get("bq_table_candidates"):
        return True
    plan = result.get("bq_sql_plan")
    if isinstance(plan, dict) and plan and not plan.get("skip_bq"):
        return True
    return False


def _render_sql_debug_meta(entry: dict[str, Any]) -> None:
    """Show per-attempt SQL metadata in the inspector."""
    bits: list[str] = []
    for key, label in (
        ("sql_source", "source"),
        ("nl2sql_model", "model"),
        ("template", "template"),
        ("pattern", "pattern"),
        ("subquestion_id", "slot"),
    ):
        val = entry.get(key)
        if val:
            bits.append(f"{label}={val}")
    if bits:
        st.caption(" · ".join(bits))


def _render_bq_sql_plan_summary(plan: dict[str, Any]) -> None:
    st.caption("Reasoner plan")
    summary: dict[str, Any] = {
        "selected_tables": plan.get("selected_tables"),
        "skip_bq": plan.get("skip_bq"),
        "rationale": plan.get("rationale"),
        "slot_path": plan.get("slot_path"),
        "reasoner_job": plan.get("reasoner_job"),
    }
    intents = plan.get("query_intents")
    if isinstance(intents, list) and intents:
        summary["query_intents"] = [
            {
                "goal": i.get("goal"),
                "tables": i.get("tables"),
                "pattern": i.get("pattern"),
                "subquestion_id": i.get("subquestion_id"),
                "measure": i.get("measure"),
            }
            for i in intents
            if isinstance(i, dict)
        ]
    st.json(summary)


def render_sql_panel(result: dict[str, Any]) -> None:
    """Always show BQ SQL / failure diagnostics in the pipeline inspector when BQ ran."""
    if not _bq_was_attempted(result):
        return

    debug_rows = [d for d in (result.get("bq_sql_debug") or []) if isinstance(d, dict)]
    bq_sql_list = list(result.get("bq_sql_queries") or [])
    if not bq_sql_list:
        seen_sql: set[str] = set()
        for row in result.get("bq_results") or []:
            if not isinstance(row, dict):
                continue
            s = str((row.get("metadata") or {}).get("sql") or "").strip()
            if s and s not in seen_sql:
                seen_sql.add(s)
                bq_sql_list.append(s)
        for d in debug_rows:
            s = str(d.get("sql") or "").strip()
            if s and s not in seen_sql:
                seen_sql.add(s)
                bq_sql_list.append(s)

    title = "BQ SQL"
    if debug_rows:
        title = f"BQ SQL ({len(debug_rows)} attempt(s))"
    elif bq_sql_list:
        title = f"BQ SQL ({len(bq_sql_list)})"
    else:
        title = "BQ SQL (none generated)"

    with st.expander(title, expanded=True):
        if show_sql_debug():
            st.caption("Inspector always shows BQ SQL attempts. RAG_SHOW_SQL_DEBUG also gates public answer SQL.")
        sql_source = str(result.get("sql_source") or "").strip()
        cache_hit = result.get("bq_cache_hit")
        if sql_source or cache_hit is not None:
            bits = []
            if sql_source:
                bits.append(f"sql_source={sql_source}")
            if cache_hit is True:
                bits.append("cache_hit=true")
            st.caption(" · ".join(bits))
        if debug_rows:
            for i, entry in enumerate(debug_rows, start=1):
                status = str(entry.get("status") or "unknown")
                st.caption(f"Attempt {i} — status={status}")
                _render_sql_debug_meta(entry)
                sql = str(entry.get("sql") or "").strip()
                if sql:
                    st.code(sql, language="sql")
                else:
                    st.warning("No SQL generated for this attempt.")
                raw = str(entry.get("nl2sql_raw") or "").strip()
                if raw:
                    with st.expander("NL2SQL raw output", expanded=False):
                        st.code(raw, language="text")
                prep = entry.get("prep_error")
                if prep:
                    st.error(f"prep_error: {prep}")
                exec_err = entry.get("execution_error")
                if exec_err:
                    st.error(f"execution_error: {exec_err}")
        elif bq_sql_list:
            for i, sql in enumerate(bq_sql_list, start=1):
                st.caption(f"Query {i}")
                st.code(sql, language="sql")
        else:
            raw_plan = result.get("bq_sql_plan")
            if isinstance(raw_plan, dict):
                plan = dict(raw_plan)
            else:
                plan = {}
            if plan.get("skip_bq"):
                st.info(f"BQ skipped by reasoner: {plan.get('rationale') or 'skip_bq'}")
            else:
                st.warning(
                    "BQ was attempted but no SQL was recorded. "
                    "Check BQ_PROJECT, NL2SQL LLM, and bq_sql_plan in raw JSON."
                )
            if plan:
                _render_bq_sql_plan_summary(plan)

def render_retrieval_tabs(result: dict[str, Any]) -> None:
    reranked = list(result.get("reranked_context") or [])
    usable_for_gen = usable_context_after_geo_purity(
        reranked,
        result.get("decomposition") if isinstance(result.get("decomposition"), dict) else None,
    )
    citations = list(result.get("citations") or [])
    tabs = st.tabs([
        f"News ({len(result.get('vector_news_results') or [])})",
        f"Academic ({len(result.get('vector_academic_papers_results') or [])})",
        f"Policy ({len(result.get('vector_policies_results') or [])})",
        f"Public reports ({len(result.get('vector_public_reports_results') or [])})",
        f"Formation ({len(result.get('vector_formation_results') or [])})",
        f"OTA ({len(result.get('vector_ota_results') or [])})",
        f"BQ tables ({len(result.get('bq_table_candidates') or [])})",
        f"BQ rows ({len(result.get('bq_results') or [])})",
        f"Merged ({len(result.get('merged_context') or [])})",
        f"Reranked ({len(reranked)})",
        f"Web ({len(result.get('web_results') or [])})",
    ])
    with tabs[0]:
        render_chunk_rows(list(result.get("vector_news_results") or []))
    with tabs[1]:
        render_chunk_rows(list(result.get("vector_academic_papers_results") or []))
    with tabs[2]:
        render_chunk_rows(list(result.get("vector_policies_results") or []))
    with tabs[3]:
        render_chunk_rows(list(result.get("vector_public_reports_results") or []))
    with tabs[4]:
        render_chunk_rows(list(result.get("vector_formation_results") or []))
    with tabs[5]:
        render_chunk_rows(list(result.get("vector_ota_results") or []))
    with tabs[6]:
        render_chunk_rows(list(result.get("bq_table_candidates") or []))
    with tabs[7]:
        render_chunk_rows(list(result.get("bq_results") or []))
    with tabs[8]:
        render_chunk_rows(list(result.get("merged_context") or []))
    with tabs[9]:
        st.caption(
            f"Reranked toward generator: {len(reranked)} · "
            f"Usable after filters: {len(usable_for_gen)} · "
            f"Cited in answer: {len(citations)}"
        )
        if len(usable_for_gen) < len(reranked):
            st.warning(
                "Geo purity or unusable-item filters removed chunks before the LLM. "
                "Gap answers with ACF no-evidence often mean this count hit zero."
            )
        render_chunk_rows(reranked)
    with tabs[10]:
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
