"""
FastAPI app for the RAG pipeline. This is the production API surface for AskADZA and other clients.

Production (GCE/Docker): Provide all secrets and configuration via environment variables
or mounted secret files at container startup. The service is 12-factor compliant and does
not require any .env files to be present or mounted in the production image.

Local dev: `load_rag_dotenv` will pick up ml-eng/config/.env and ml-eng/data/local/.env
(with standard precedence and force-key rules).

Run locally: uvicorn ml.rag.api:app --reload --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

# Use the single, authoritative environment loader for the entire RAG stack.
# This ensures config/.env (and RAG_* vars) are respected, and the service works
# in pure 12-factor mode when only env vars are supplied (the GCE production path).
from ml.rag.local_env import load_rag_dotenv

# ml-eng/ directory is parents[3] from ml/rag/app/api.py
_ml_eng = Path(__file__).resolve().parents[3]
load_rag_dotenv(_ml_eng)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import logging

from ml.rag.chat_history import normalize_messages

logger = logging.getLogger("ml.rag.api")
from ml.rag.chat_memory import append_turn_and_compact, flat_messages_to_memory
from ml.rag.session_store import get_session_blob, save_session_blob, redis_status
from ml.rag.observability import create_trace

app = FastAPI(
    title="OpenTrace RAG API",
    description="Query BigQuery + vector DB via a graph RAG; use from the frontend chatbot.",
    version="0.1.0",
)

# Allow frontend to call from another origin (set RAG_CORS_ORIGINS for production)
_cors_origins = os.environ.get("RAG_CORS_ORIGINS", "*").strip().split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question for the RAG")
    include_trace: bool = Field(False, description="Include decomposition and retrieval counts in response")
    session_id: str | None = Field(
        None,
        description="Omit to start a new session; reuse for multi-turn chat (server-side memory)",
    )
    conversation_history: list[ChatMessage] | None = Field(
        None,
        description="If set, used as prior turns instead of server session store for this request",
    )
    geo_override: str | None = None
    time_start_override: str | None = None
    time_end_override: str | None = None
    news_top_k: int | None = None
    academic_top_k: int | None = None
    bq_top_k: int | None = None
    rerank_top_k: int | None = None
    ota_top_k: int | None = None

    # --- AskADZA / client-driven audience support (client owns profile & tone) ---
    stakeholder_type: str | None = Field(
        None,
        description="Optional audience/persona identifier (e.g. government_public, private_sector, farmers_communities). The AskADZA client decides the value and meaning.",
    )
    audience_instructions: str | None = Field(
        None,
        description="Optional free-form instructions or tone guidance supplied by the client UI. If present, the RAG generator should incorporate this (client overrides server-side defaults).",
        max_length=4000,
    )


class QueryResponse(BaseModel):
    answer: str
    session_id: str = Field(..., description="Pass on the next request for chat continuity")
    error: str | None = None
    has_bq_results: bool = False
    has_vector_results: bool = False
    bq_sql: str | None = None
    trace: dict | None = None


@app.get("/health")
async def health():
    """Liveness probe for GCE / orchestrators. Always returns quickly."""
    return {"status": "ok", "service": "rag"}


@app.get("/ready")
async def ready():
    """
    Readiness probe.
    Reports whether the minimum configuration for Qdrant + LLM appears to be present
    (without revealing secrets). Useful for GCE managed instance groups / load balancers.
    Redis connectivity (if configured via RAG_REDIS_URL) is reported for observability but
    does not affect the ready status (sessions gracefully fall back to in-memory).
    """
    # Critical keys for the current production scope (News/Research + BQ via NL2SQL path)
    critical = ["QDRANT_URL", "QDRANT_API_KEY", "RAG_LLM_BASE_URL"]
    missing = [k for k in critical if not os.environ.get(k, "").strip()]

    redis_info: dict[str, Any] | None = None
    if os.environ.get("RAG_REDIS_URL") or os.environ.get("REDIS_URL"):
        try:
            redis_info = redis_status()
        except Exception:
            redis_info = {"backend": "error", "connected": False}

    payload: dict[str, Any] = {
        "status": "ready" if not missing else "not_ready",
        "service": "rag",
        "missing_config_keys": missing,
    }
    if redis_info:
        payload["redis"] = redis_info
    return payload


def _resolve_prior_memory(request: QueryRequest) -> tuple[str, str, list[dict[str, str]]]:
    """Return (session_id, conversation_summary, recent_turns) before the current user message."""
    if request.conversation_history is not None:
        sid = (request.session_id or "").strip() or uuid.uuid4().hex
        raw_msgs = [
            m.model_dump() if hasattr(m, "model_dump") else m.dict()
            for m in request.conversation_history
        ]
        prior = normalize_messages(raw_msgs)
        summary, recent = flat_messages_to_memory(prior)
        return sid, summary, recent

    sid = (request.session_id or "").strip() or uuid.uuid4().hex
    blob = get_session_blob(sid) or {"conversation_summary": "", "recent_turns": []}
    summary = str(blob.get("conversation_summary") or "")
    recent = normalize_messages(blob.get("recent_turns"))
    return sid, summary, recent


def _persist_session_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    blob = get_session_blob(session_id) or {"conversation_summary": "", "recent_turns": []}
    summary, recent = append_turn_and_compact(
        str(blob.get("conversation_summary") or ""),
        blob.get("recent_turns"),
        user_msg,
        assistant_msg,
    )
    save_session_blob(session_id, {"conversation_summary": summary, "recent_turns": recent})


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run the RAG pipeline and return the answer for the frontend chatbot."""
    try:
        logger.info("query request received (len=%d, has_history=%s, stakeholder=%s)",
                    len(request.query or ""), bool(request.conversation_history), bool(request.stakeholder_type))

        from ml.rag.graph import run_rag

        session_id, prior_summary, prior_recent = _resolve_prior_memory(request)

        # Optional Langfuse root trace (groups the LangGraph spans + LLM calls)
        create_trace(
            "rag.query",
            session_id=session_id,
            stakeholder_type=request.stakeholder_type,
            query=request.query,
            has_history=bool(request.conversation_history),
        )
        kwargs: dict = {}
        if prior_summary.strip() or prior_recent:
            kwargs["conversation_summary"] = prior_summary
            kwargs["recent_turns"] = prior_recent
        for key in (
            "geo_override",
            "time_start_override",
            "time_end_override",
            "news_top_k",
            "academic_top_k",
            "bq_top_k",
            "rerank_top_k",
            "ota_top_k",
            "stakeholder_type",
            "audience_instructions",
        ):
            val = getattr(request, key, None)
            if val is not None:
                if isinstance(val, str):
                    val = val.strip()
                if val:
                    kwargs[key] = val

        result = run_rag(request.query, **kwargs)
        # Try to extract the SQL used for BQ retrieval (if any)
        bq_sql: str | None = None
        for item in result.get("bq_results") or []:
            meta = item.get("metadata") or {}
            sql = meta.get("sql")
            if isinstance(sql, str) and sql.strip():
                bq_sql = sql.strip()
                break
        trace: dict | None = None
        if request.include_trace:
            trace = {
                "decomposition": result.get("decomposition"),
                "bq_table_candidates_count": len(result.get("bq_table_candidates") or []),
                "vector_news_count": len(result.get("vector_news_results") or []),
                "vector_academic_count": len(result.get("vector_academic_results") or []),
                "merged_context_count": len(result.get("merged_context") or []),
                "reranked_context_count": len(result.get("reranked_context") or []),
            }

        answer = result.get("answer", "") or ""
        if request.conversation_history is None:
            _persist_session_turn(session_id, request.query.strip(), answer)

        return QueryResponse(
            answer=answer,
            session_id=session_id,
            error=result.get("error"),
            has_bq_results=bool(result.get("bq_results")),
            has_vector_results=bool(result.get("vector_results")),
            bq_sql=bq_sql,
            trace=trace,
        )
    except Exception as e:
        import traceback
        detail = str(e)
        if os.environ.get("RAG_DEBUG", "").strip().lower() in ("1", "true", "on"):
            detail += "\n\n" + traceback.format_exc()
        elif "nn" in detail.lower() or "not defined" in detail.lower():
            detail += ". If using the vector retriever, install PyTorch: pip install torch"
        raise HTTPException(status_code=500, detail=detail)


@app.get("/")
async def root():
    return {"message": "OpenTrace RAG API", "docs": "/docs", "health": "/health", "query": "POST /query"}
