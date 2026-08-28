"""
Streamlit pipeline inspector for the multi-source RAG graph.

Shows full flow observability: route, decomposition, all retrieval arms, merge/rerank,
web fallback, generator input, usage, and latency.

Run: PYTHONPATH=ml-eng streamlit run ml/rag/chatbot/streamlit_app.py
"""
from __future__ import annotations

import atexit
import os
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

_ml_eng = Path(__file__).resolve().parents[3]
from ml.rag.chatbot.geo_policy import FARMER_PLAN_TYPE
from ml.rag.chatbot.plan_policy import PLAN_TYPES
from ml.rag.chatbot.stakeholder_prompts import CATEGORIES
from ml.rag.chatbot.qa_run_kwargs import build_run_kwargs
from ml.rag.chatbot.streamlit_inspector import (
    PRESET_QUERIES,
    BackendMode,
    debug_default_enabled,
    query_via_http_api,
    render_pipeline_inspector,
)
from ml.rag.observability import flush_langfuse, rag_trace_context
from ml.rag.chat_memory import append_turn_and_compact
from ml.rag.local_env import load_rag_dotenv

load_rag_dotenv(_ml_eng)
atexit.register(flush_langfuse)

st.set_page_config(page_title="OpenTrace RAG — Pipeline inspector", page_icon="🔍", layout="wide")
st.title("OpenTrace RAG — pipeline inspector (QA)")
st.caption(
    "Test the full RAG graph with retrieval, routing, and generation observability after each turn."
)


def _session_label(sid: str) -> str:
    sess = st.session_state.rag_sessions.get(sid) or {}
    msgs = sess.get("messages") or []
    if not msgs:
        return f"{sid[:8]}… (empty)"
    for m in msgs:
        if m.get("role") == "user":
            t = (m.get("content") or "").strip().replace("\n", " ")
            return (t[:40] + "…") if len(t) > 40 else t
    return f"{sid[:8]}…"


def _ensure_sessions() -> None:
    if "rag_sessions" not in st.session_state:
        sid = uuid.uuid4().hex
        st.session_state.rag_sessions = {sid: {"messages": []}}
        st.session_state.active_session_id = sid
        st.session_state.api_session_id = None
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = next(iter(st.session_state.rag_sessions))
    if "api_session_id" not in st.session_state:
        st.session_state.api_session_id = None


def _new_chat() -> None:
    sid = uuid.uuid4().hex
    st.session_state.rag_sessions[sid] = {"messages": []}
    st.session_state.active_session_id = sid
    st.session_state.api_session_id = None


def _delete_active_session() -> None:
    opts = list(st.session_state.rag_sessions.keys())
    cur = st.session_state.active_session_id
    if len(opts) <= 1:
        st.session_state.rag_sessions = {uuid.uuid4().hex: {"messages": []}}
        st.session_state.active_session_id = next(iter(st.session_state.rag_sessions))
        st.session_state.api_session_id = None
        return
    del st.session_state.rag_sessions[cur]
    st.session_state.active_session_id = opts[0] if opts[0] != cur else opts[1]
    st.session_state.api_session_id = None


def _run_pipeline(
    prompt: str,
    kwargs: dict[str, Any],
    *,
    backend_mode: BackendMode,
    api_base_url: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if backend_mode == "http_api":
        base = api_base_url.strip()
        if not base:
            raise ValueError("Set RAG_API_BASE_URL or enter an API base URL for HTTP mode.")
        result = query_via_http_api(
            base,
            prompt.strip(),
            kwargs=kwargs,
            session_id=st.session_state.active_session_id,
        )
        sid = result.get("session_id")
        if isinstance(sid, str) and sid.strip():
            st.session_state.api_session_id = sid.strip()
            st.session_state.active_session_id = sid.strip()
    else:
        from ml.rag.graph import run_rag

        session_id = st.session_state.active_session_id
        kwargs["session_id"] = session_id
        kwargs["trace_tags"] = ["streamlit-qa"]
        with rag_trace_context(
            trace_name="rag.streamlit",
            session_id=session_id,
            plan_type=str(kwargs.get("plan_type") or ""),
            category=str(kwargs.get("category") or ""),
            trace_input={"query": prompt.strip()[:500]},
            tags=["streamlit-qa"],
        ) as trace_handle:
            result = run_rag(prompt.strip(), **kwargs)
            trace_handle.update_output(result, latency_ms=result.get("latency_ms"))
        flush_langfuse()
        result["_backend_mode"] = "in_process"
        result["_query"] = prompt.strip()
    result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    return result


_ensure_sessions()

with st.sidebar:
    st.subheader("Chat sessions")
    opts = list(st.session_state.rag_sessions.keys())
    if st.session_state.active_session_id not in opts:
        st.session_state.active_session_id = opts[0]
    ix = opts.index(st.session_state.active_session_id)
    chosen = st.selectbox(
        "Active session",
        range(len(opts)),
        index=ix,
        format_func=lambda i: _session_label(opts[i]),
    )
    st.session_state.active_session_id = opts[chosen]
    c1, c2 = st.columns(2)
    with c1:
        if st.button("New chat"):
            _new_chat()
            st.rerun()
    with c2:
        if st.button("Delete session"):
            _delete_active_session()
            st.rerun()

    st.divider()
    st.subheader("Backend")
    backend_labels = ["In-process (full detail)", "HTTP API (counts only)"]
    backend_choice = st.radio(
        "Execution mode",
        backend_labels,
        index=0,
        help="In-process calls run_rag() locally. HTTP mode hits POST /query on Railway or local API.",
    )
    backend_mode: BackendMode = "http_api" if backend_choice == backend_labels[1] else "in_process"
    default_api = os.environ.get("RAG_API_BASE_URL", "").strip()
    api_base_url = st.text_input(
        "API base URL (HTTP mode)",
        value=default_api,
        placeholder="https://your-railway-service.up.railway.app",
    )

    st.divider()
    st.subheader("Preset queries")
    for label, preset_query, preset_kwargs in PRESET_QUERIES:
        if st.button(label, key=f"preset_{label}"):
            st.session_state.pending_prompt = preset_query
            st.session_state.pending_preset_kwargs = preset_kwargs
            st.rerun()

    st.divider()
    st.subheader("LLM backend")
    from ml.rag.llm_chat import llm_chat_completions_url, llm_configured, llm_model_id

    st.caption(f"URL: {llm_chat_completions_url() or '(not configured)'}")
    st.caption(f"Model: {llm_model_id()}")
    st.caption(f"Configured: {llm_configured()}")
    _reranker_mode_env = os.environ.get("RAG_RERANKER_MODE", "").strip().lower()
    _cohere_key = (
        os.environ.get("RAG_RERANKER_COHERE_API_KEY")
        or os.environ.get("COHERE_API_KEY")
        or ""
    ).strip()
    _legacy_rerank = os.environ.get("RAG_LLM_RERANK", "").strip().lower()
    if _reranker_mode_env in {"cohere", "cross_encoder", "llm", "off"}:
        reranker_mode = _reranker_mode_env
    elif _cohere_key:
        reranker_mode = "cohere"
    elif _legacy_rerank in {"on", "1", "true", "yes"}:
        reranker_mode = "llm"
    elif _legacy_rerank in {"off", "0", "false", "no"}:
        reranker_mode = "off"
    else:
        reranker_mode = "cross_encoder"
    st.caption(f"Reranker: {reranker_mode}")
    if not llm_configured() and backend_mode == "in_process":
        st.warning("Set RAG_LLM_BASE_URL in ml-eng/config/.env and restart Streamlit.")

    st.divider()
    st.subheader("Retrieval controls")
    news_top_k = st.number_input("News chunks (top_k)", min_value=1, max_value=50, value=12)
    academic_top_k = st.number_input("Academic chunks (top_k)", min_value=1, max_value=50, value=10)
    bq_top_k = st.number_input("BQ rows (top_k)", min_value=1, max_value=100, value=12)
    ota_top_k = st.number_input("OTA chunks (top_k)", min_value=1, max_value=30, value=10)
    rerank_top_k = st.number_input("Rerank context size", min_value=1, max_value=50, value=18)
    st.divider()
    plan_type_options = [p["id"] for p in PLAN_TYPES]
    category_options = [c["id"] for c in CATEGORIES]
    _default_plan = "Farmers"
    _default_category = "Farmers"
    _plan_ix = plan_type_options.index(_default_plan) if _default_plan in plan_type_options else 0
    _cat_ix = category_options.index(_default_category) if _default_category in category_options else 0

    def _catalog_label(catalog: list[dict[str, str]], item_id: str) -> str:
        for item in catalog:
            if item["id"] == item_id:
                return str(item["label"])
        return item_id

    plan_type = st.selectbox(
        "Plan type",
        plan_type_options,
        index=_plan_ix,
        format_func=lambda x: _catalog_label(PLAN_TYPES, x),
    )
    category = st.selectbox(
        "Category (audience lens)",
        category_options,
        index=_cat_ix,
        format_func=lambda x: _catalog_label(CATEGORIES, x),
    )
    profile_country = ""
    if plan_type == FARMER_PLAN_TYPE:
        profile_country = st.text_input(
            "Profile country (Farmers plan only)",
            placeholder="e.g. Nigeria",
            help="Used as retrieval geo filter when plan_type is Farmers.",
        )
    t_start = st.text_input("Time start YYYY-MM-DD (optional)", placeholder="2020-01-01")
    t_end = st.text_input("Time end YYYY-MM-DD (optional)", placeholder="2025-12-31")
    show_debug = st.checkbox("Show pipeline inspector (last run)", value=debug_default_enabled())

active = st.session_state.active_session_id
sess = st.session_state.rag_sessions[active]
if "conversation_summary" not in sess:
    sess["conversation_summary"] = ""
if "recent_turns" not in sess:
    sess["recent_turns"] = []
messages: list[dict[str, str]] = sess["messages"]
prior_summary = str(sess.get("conversation_summary") or "")
prior_recent = list(sess.get("recent_turns") or [])

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m.get("content") or "")

prompt = st.chat_input("Ask a question…")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = str(st.session_state.pop("pending_prompt"))
preset_overrides = st.session_state.pop("pending_preset_kwargs", None)

if prompt:
    kwargs = build_run_kwargs(
        news_top_k=int(news_top_k),
        academic_top_k=int(academic_top_k),
        bq_top_k=int(bq_top_k),
        ota_top_k=int(ota_top_k),
        rerank_top_k=int(rerank_top_k),
        plan_type=plan_type,
        category=category,
        profile_country=profile_country,
        t_start=t_start,
        t_end=t_end,
        prior_summary=prior_summary,
        prior_recent=prior_recent,
        preset_overrides=preset_overrides if isinstance(preset_overrides, dict) else None,
    )

    with st.spinner("Running pipeline…"):
        try:
            result = _run_pipeline(
                prompt,
                kwargs,
                backend_mode=backend_mode,
                api_base_url=api_base_url,
            )
            answer = result.get("answer") or ""
            citations = result.get("citations") or []
            err = result.get("error")
            if err:
                answer = f"**Error:** {err}\n\n{answer}".strip()
            if citations and "Sources" not in answer:
                cite_lines = [
                    f"{c.get('id')}. {c.get('text')}"
                    for c in citations
                    if isinstance(c, dict)
                ]
                if cite_lines:
                    answer = (answer.rstrip() + "\n\nSources\n" + "\n".join(cite_lines)).strip()

            messages.append({"role": "user", "content": prompt.strip()})
            messages.append({"role": "assistant", "content": answer})
            if backend_mode == "in_process":
                new_summary, new_recent = append_turn_and_compact(
                    prior_summary,
                    prior_recent,
                    prompt.strip(),
                    answer,
                )
                sess["conversation_summary"] = new_summary
                sess["recent_turns"] = new_recent
            if show_debug:
                st.session_state.last_rag_debug = result
                st.session_state.last_rag_debug_meta = {
                    "query": prompt.strip(),
                    "memory_summary_len": len(sess.get("conversation_summary") or ""),
                    "memory_recent_count": len(sess.get("recent_turns") or []),
                    "backend_mode": backend_mode,
                }
            st.rerun()
        except Exception as e:
            st.exception(e)

if show_debug and "last_rag_debug" in st.session_state:
    meta = st.session_state.get("last_rag_debug_meta") or {}
    render_pipeline_inspector(
        st.session_state.last_rag_debug,
        latency_ms=st.session_state.last_rag_debug.get("latency_ms"),
        backend_mode=meta.get("backend_mode") or backend_mode,
        query=str(meta.get("query") or ""),
        memory_summary_len=int(meta.get("memory_summary_len") or 0),
        memory_recent_count=int(meta.get("memory_recent_count") or 0),
    )

st.divider()
st.markdown("**CLI**")
st.code(
    "PYTHONPATH=ml-eng streamlit run ml/rag/chatbot/streamlit_app.py",
    language="bash",
)
