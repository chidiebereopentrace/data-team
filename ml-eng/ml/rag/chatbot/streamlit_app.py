"""
Streamlit test UI for the multi-source RAG pipeline:
decompose → parallel retrieval (BQ descriptions + news + academic) → BQ → merge → rerank → generate.

Multi-turn chat: sessions in st.session_state; prior turns passed to the generator (retrieval uses latest message only).

Run from repo root: streamlit run ml/rag/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

# Load env: data/local/.env then config/.env (BQ + LLM keys not duplicated in local)
# streamlit_app.py lives at ml-eng/ml/rag/chatbot/, so parents[3] is `ml-eng/` (load_rag_dotenv needs this).
_ml_eng = Path(__file__).resolve().parents[3]
from ml.rag.local_env import load_rag_dotenv

# #region agent log
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]  # data-team workspace root


def _agent_debug_log_runtime(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": "6c8b2f",
            "id": f"log_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with (_WORKSPACE_ROOT / "debug-6c8b2f.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


_agent_debug_log_runtime(
    "streamlit_app.py:env_probe:before",
    "about to load rag dotenv",
    {
        "ml_eng_root": str(_ml_eng),
        "cwd": str(Path.cwd()),
        "config_env_exists": (_ml_eng / "config" / ".env").is_file(),
        "data_local_env_exists": (_ml_eng / "data" / "local" / ".env").is_file(),
    },
    "A",
)
# #endregion

load_rag_dotenv(_ml_eng)

# #region agent log
_agent_debug_log_runtime(
    "streamlit_app.py:env_probe:after",
    "loaded rag dotenv",
    {
        "qdrant_url_present": bool(os.environ.get("QDRANT_URL", "").strip()),
        "qdrant_api_key_present": bool(os.environ.get("QDRANT_API_KEY", "").strip()),
        "qdrant_url_len": len(os.environ.get("QDRANT_URL", "")),
        "qdrant_api_key_len": len(os.environ.get("QDRANT_API_KEY", "")),
        "rag_llm_base_url": os.environ.get("RAG_LLM_BASE_URL", ""),
        "rag_llm_model_id": os.environ.get("RAG_LLM_MODEL_ID", ""),
    },
    "D",
)


def _probe_qdrant_collection_dims() -> None:
    """Introspect all RAG collections to compare ingest-time vs query-time dims (H-E/F/G)."""
    try:
        from qdrant_client import QdrantClient

        url = os.environ.get("QDRANT_URL", "").strip().strip('"').strip("'")
        api_key = os.environ.get("QDRANT_API_KEY", "").strip().strip('"').strip("'")
        if not url or not api_key:
            return
        client = QdrantClient(url=url, api_key=api_key, check_compatibility=False, timeout=30)
        targets = [
            ("news_data", os.environ.get("QDRANT_COLLECTION_NEWS", "news_data")),
            ("research_other_papers", os.environ.get("QDRANT_COLLECTION_RESEARCH_PAPERS", "research_other_papers")),
            ("BQ_table_descriptions", os.environ.get("QDRANT_COLLECTION_DATA_DESCRIPTIONS", "BQ_table_descriptions")),
            ("OTA_insights", os.environ.get("QDRANT_COLLECTION_OTA_INSIGHTS", "OTA_insights")),
        ]
        for label, name in targets:
            try:
                info = client.get_collection(collection_name=name)
                params = getattr(info.config, "params", None)
                vectors = getattr(params, "vectors", None) if params is not None else None
                sparse = getattr(params, "sparse_vectors", None) if params is not None else None
                # vectors may be either a single VectorParams or dict[name->VectorParams]
                dense_summary: dict[str, Any] = {}
                if hasattr(vectors, "size"):
                    dense_summary["<unnamed>"] = {"size": getattr(vectors, "size", None), "distance": str(getattr(vectors, "distance", ""))}
                elif isinstance(vectors, dict):
                    for vname, vparams in vectors.items():
                        dense_summary[vname] = {
                            "size": getattr(vparams, "size", None),
                            "distance": str(getattr(vparams, "distance", "")),
                        }
                sparse_names = list(sparse.keys()) if isinstance(sparse, dict) else []
                _agent_debug_log_runtime(
                    "streamlit_app.py:qdrant_collection_probe",
                    f"collection dims for {label}",
                    {
                        "collection": name,
                        "points_count": getattr(info, "points_count", None),
                        "dense_vectors": dense_summary,
                        "sparse_vector_names": sparse_names,
                    },
                    "E",
                )
            except Exception as exc:
                _agent_debug_log_runtime(
                    "streamlit_app.py:qdrant_collection_probe",
                    f"collection probe failed for {label}",
                    {"collection": name, "error": str(exc)[:200]},
                    "E",
                )
    except Exception as exc:
        _agent_debug_log_runtime(
            "streamlit_app.py:qdrant_collection_probe",
            "client init failed",
            {"error": str(exc)[:200]},
            "E",
        )


_probe_qdrant_collection_dims()
# #endregion

import streamlit as st

# #region agent log
def _agent_debug_log(message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix") -> None:
    import json
    import time

    _p = Path(__file__).resolve().parents[2] / ".cursor" / "debug-4fd6d6.log"
    try:
        _p.parent.mkdir(parents=True, exist_ok=True)
        with _p.open("a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "4fd6d6",
                        "timestamp": int(time.time() * 1000),
                        "location": "streamlit_app.py:startup",
                        "message": message,
                        "data": data,
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                    }
                )
                + "\n"
            )
    except OSError:
        pass


try:
    import importlib.util as _ilu

    _tv = _ilu.find_spec("torchvision") is not None
except Exception:
    _tv = False
_agent_debug_log("import_probe", {"torchvision_spec_found": _tv}, "A")

try:
    import sys as _sys

    _agent_debug_log(
        "streamlit_runtime_probe",
        {
            "streamlit_version": getattr(st, "__version__", None),
            # Note: Streamlit executes the app in a separate runtime where argv may not include CLI flags.
            "argv": list(getattr(_sys, "argv", [])[:12]),
            "env_STREAMLIT_SERVER_FILE_WATCHER_TYPE": os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE"),
        },
        "B",
    )
except Exception:
    pass

try:
    # Use the public Streamlit API; streamlit.config is an internal module not in type stubs.
    _agent_debug_log(
        "streamlit_config_probe",
        {
            "server.fileWatcherType": st.get_option("server.fileWatcherType"),
        },
        "B",
    )
except Exception:
    pass

try:
    # Hypothesis E: container exits with 137 due to SIGTERM/SIGINT (user stop) vs SIGKILL (OOM).
    import signal as _signal

    def _agent_signal_handler(signum, _frame):  # type: ignore[no-untyped-def]
        _agent_debug_log("signal_received", {"signum": int(signum)}, "E")

    _signal.signal(_signal.SIGTERM, _agent_signal_handler)
    _signal.signal(_signal.SIGINT, _agent_signal_handler)
except Exception:
    pass

try:
    # Hypothesis F: memory pressure / cgroup limits (OOM kill → SIGKILL, no handler).
    def _read_text(p: str) -> str | None:
        try:
            with open(p, "r", encoding="utf-8") as _f:
                return _f.read().strip()
        except OSError:
            return None

    _status = _read_text("/proc/self/status") or ""
    _mem_lines = [ln for ln in _status.splitlines() if ln.startswith(("VmRSS:", "VmHWM:", "VmSize:"))]
    _agent_debug_log(
        "mem_probe",
        {
            "proc_status_mem": _mem_lines[:6],
            "cgroup_memory_max": _read_text("/sys/fs/cgroup/memory.max"),
            "cgroup_memory_current": _read_text("/sys/fs/cgroup/memory.current"),
        },
        "F",
    )
except Exception:
    pass
# #endregion

from ml.rag.chat_memory import append_turn_and_compact

st.set_page_config(page_title="OpenTrace RAG (test)", page_icon="🔍", layout="wide")
st.title("OpenTrace RAG — test interface")
st.caption(
    "Advisory answers from retrieved news, research, and OpenTrace structured data — grounded in sources only."
)


def _show_sql_debug() -> bool:
    return os.environ.get("RAG_SHOW_SQL_DEBUG", "").strip().lower() in ("1", "true", "on", "yes")


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
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = next(iter(st.session_state.rag_sessions))


def _new_chat() -> None:
    sid = uuid.uuid4().hex
    st.session_state.rag_sessions[sid] = {"messages": []}
    st.session_state.active_session_id = sid


def _delete_active_session() -> None:
    opts = list(st.session_state.rag_sessions.keys())
    cur = st.session_state.active_session_id
    if len(opts) <= 1:
        st.session_state.rag_sessions = {uuid.uuid4().hex: {"messages": []}}
        st.session_state.active_session_id = next(iter(st.session_state.rag_sessions))
        return
    del st.session_state.rag_sessions[cur]
    st.session_state.active_session_id = opts[0] if opts[0] != cur else opts[1]


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
    st.subheader("LLM backend")
    from ml.rag.llm_chat import llm_chat_completions_url, llm_configured, llm_model_id

    llm_url = llm_chat_completions_url() or "(not configured)"
    st.caption(f"URL: {llm_url}")
    st.caption(f"Model: {llm_model_id()}")
    st.caption(f"Configured: {llm_configured()}")
    rerank_on = os.environ.get("RAG_LLM_RERANK", "off").strip().lower() not in ("off", "0", "false")
    st.caption(f"LLM rerank: {'on' if rerank_on else 'off (pass-through order)'}")
    if not llm_configured():
        st.warning("Set RAG_LLM_BASE_URL in ml-eng/config/.env and restart Streamlit.")

    st.divider()
    st.subheader("Retrieval controls")
    news_top_k = st.number_input("News chunks (top_k)", min_value=1, max_value=50, value=20)
    academic_top_k = st.number_input("Academic chunks (top_k)", min_value=1, max_value=50, value=20)
    bq_top_k = st.number_input("BQ rows (top_k)", min_value=1, max_value=100, value=15)
    ota_top_k = st.number_input("OTA chunks (top_k) — local test only", min_value=1, max_value=30, value=10)
    rerank_top_k = st.number_input("Rerank context size", min_value=1, max_value=50, value=20)
    st.divider()
    from ml.rag.chatbot.geo_policy import FARMER_STAKEHOLDER
    from ml.rag.chatbot.stakeholder_prompts import STAKEHOLDER_TYPES

    stakeholder_options = [""] + [s["id"] for s in STAKEHOLDER_TYPES]
    stakeholder_type = st.selectbox(
        "Stakeholder type (optional)",
        stakeholder_options,
        format_func=lambda x: x or "(none)",
    )
    profile_country = ""
    if stakeholder_type == FARMER_STAKEHOLDER:
        profile_country = st.text_input(
            "Profile country (farmers only)",
            placeholder="e.g. Nigeria",
            help="Used as retrieval geo filter for farmers_communities only.",
        )
    t_start = st.text_input("Time start YYYY-MM-DD (optional)", placeholder="2020-01-01")
    t_end = st.text_input("Time end YYYY-MM-DD (optional)", placeholder="2025-12-31")
    show_debug = st.checkbox("Show pipeline debug (last run)", value=False)

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

if prompt:
    kwargs: dict = {
        "news_top_k": int(news_top_k),
        "academic_top_k": int(academic_top_k),
        "bq_top_k": int(bq_top_k),
        "ota_top_k": int(ota_top_k),
        "rerank_top_k": int(rerank_top_k),
    }
    if stakeholder_type:
        kwargs["stakeholder_type"] = stakeholder_type
    if profile_country.strip():
        kwargs["user_profile"] = {"country": profile_country.strip()}
    if t_start.strip():
        kwargs["time_start_override"] = t_start.strip()[:10]
    if t_end.strip():
        kwargs["time_end_override"] = t_end.strip()[:10]
    if prior_summary.strip() or prior_recent:
        kwargs["conversation_summary"] = prior_summary
        kwargs["recent_turns"] = prior_recent

    with st.spinner("Running pipeline…"):
        try:
            from ml.rag.graph import run_rag

            result = run_rag(prompt.strip(), **kwargs)
            answer = result.get("answer") or ""
            citations = result.get("citations") or []
            err = result.get("error")
            if err:
                answer = f"**Error:** {err}\n\n{answer}".strip()
            if citations and "Sources" not in answer:
                cite_lines = [f"{c.get('id')}. {c.get('text')}" for c in citations if isinstance(c, dict)]
                if cite_lines:
                    answer = (answer.rstrip() + "\n\nSources\n" + "\n".join(cite_lines)).strip()

            messages.append({"role": "user", "content": prompt.strip()})
            messages.append({"role": "assistant", "content": answer})
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
            st.rerun()
        except Exception as e:
            st.exception(e)

def _render_chunk_rows(items: list[dict[str, Any]], *, preview_chars: int = 600) -> None:
    """Render retrieval/rerank chunks as collapsible rows with score, metadata, and content preview."""
    if not items:
        st.info("No items.")
        return
    for i, it in enumerate(items, start=1):
        content = str(it.get("content") or "")
        score = it.get("score")
        raw_meta = it.get("metadata")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        source = it.get("source") or it.get("_context_kind") or "?"
        title_bits = [f"#{i}", f"[{source}]"]
        if isinstance(score, (int, float)):
            title_bits.append(f"score={score:.4f}")
        # Helpful metadata: title/source_file/authors/doi/url/country/date.
        for k in ("section_title", "label", "source_file", "title", "url", "doi", "table_name", "geo_country_primary", "country", "published_at"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                title_bits.append(f"{k}={v.strip()[:50]}")
                break
        header = " · ".join(title_bits)
        with st.expander(header, expanded=False):
            if isinstance(score, (int, float)):
                st.caption(f"score: {score:.6f}  ·  source: {source}")
            if meta:
                st.json({k: v for k, v in meta.items() if v is not None and v != ""}, expanded=False)
            preview = content if len(content) <= preview_chars else content[:preview_chars] + "…"
            st.markdown(preview if preview else "_(empty content)_")


if show_debug and "last_rag_debug" in st.session_state:
    result = st.session_state.last_rag_debug
    with st.expander("Pipeline debug (last run)", expanded=True):
        dec = result.get("decomposition") or {}
        st.subheader("Query decomposition")
        st.json(dec)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("BQ table-description matches", len(result.get("bq_table_candidates") or []))
        with c2:
            st.metric("Structured data rows", len(result.get("bq_results") or []))
        with c3:
            st.metric("News chunks", len(result.get("vector_news_results") or []))
        with c4:
            st.metric("Research corpus chunks", len(result.get("vector_academic_results") or []))
        with c5:
            st.metric("Reranked → generator", len(result.get("reranked_context") or []))

        if _show_sql_debug():
            bq_sql_list = result.get("bq_sql_queries") or []
            if not bq_sql_list:
                bq_rows = result.get("bq_results") or []
                seen_sql: set[str] = set()
                for row in bq_rows:
                    s = str((row.get("metadata") or {}).get("sql") or "").strip()
                    if s and s not in seen_sql:
                        seen_sql.add(s)
                        bq_sql_list.append(s)
            if bq_sql_list:
                with st.expander(f"Internal: generated SQL ({len(bq_sql_list)})", expanded=False):
                    st.caption("Not shown to end users. Set RAG_SHOW_SQL_DEBUG=0 to hide.")
                    for i, sql in enumerate(bq_sql_list, start=1):
                        st.caption(f"Query {i}")
                        st.code(sql, language="sql")

        st.subheader("Retrieved from each collection")
        tab_news, tab_research, tab_bq_desc, tab_bq_rows, tab_merged, tab_used = st.tabs([
            f"News ({len(result.get('vector_news_results') or [])})",
            f"Research / Policy / Public Report ({len(result.get('vector_academic_results') or [])})",
            f"BQ table descriptions ({len(result.get('bq_table_candidates') or [])})",
            f"Structured data ({len(result.get('bq_results') or [])})",
            f"Merged before rerank ({len(result.get('merged_context') or [])})",
            f"Passed to generator ({len(result.get('reranked_context') or [])})",
        ])
        with tab_news:
            _render_chunk_rows(list(result.get("vector_news_results") or []))
        with tab_research:
            _render_chunk_rows(list(result.get("vector_academic_results") or []))
        with tab_bq_desc:
            _render_chunk_rows(list(result.get("bq_table_candidates") or []))
        with tab_bq_rows:
            _render_chunk_rows(list(result.get("bq_results") or []))
        with tab_merged:
            _render_chunk_rows(list(result.get("merged_context") or []))
        with tab_used:
            st.caption("Items in this tab are the exact context block the generator (LLM) saw, in order.")
            _render_chunk_rows(list(result.get("reranked_context") or []))

st.divider()
st.markdown("**CLI**")
st.code(
    "PYTHONPATH=. python -m ml/rag.run \"Your question\"\n"
    "streamlit run ml/rag/streamlit_app.py",
    language="bash",
)
