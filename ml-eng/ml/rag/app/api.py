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

from ml.rag.api_schemas import CitationItem, UsageStats, UserProfile
from ml.rag.chat_history import normalize_messages
from ml.rag.request_context import resolve_request_context

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
    chat_history: list[ChatMessage] | None = Field(
        None,
        description="Prior turns for this request (canonical). Server session is not updated when set.",
    )
    conversation_history: list[ChatMessage] | None = Field(
        None,
        description="Deprecated alias for chat_history.",
    )
    geo_override: str | None = Field(
        None,
        description="Deprecated and ignored. Use user_profile.country for farmers_communities retrieval geo.",
    )
    user_profile: UserProfile | None = Field(
        None,
        description="User profile: country (farmers retrieval geo), stakeholder_type, audience_instructions.",
    )
    time_start_override: str | None = None
    time_end_override: str | None = None
    news_top_k: int | None = None
    academic_top_k: int | None = None
    bq_top_k: int | None = None
    rerank_top_k: int | None = None
    ota_top_k: int | None = None

    # Deprecated: prefer user_profile.stakeholder_type / user_profile.audience_instructions
    stakeholder_type: str | None = Field(
        None,
        description="Deprecated. Use user_profile.stakeholder_type.",
    )
    audience_instructions: str | None = Field(
        None,
        description="Deprecated. Use user_profile.audience_instructions.",
        max_length=4000,
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    session_id: str = Field(..., description="Pass on the next request for chat continuity")
    usage: UsageStats = Field(default_factory=lambda: UsageStats())
    error: str | None = None
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
    critical = ["QDRANT_URL", "QDRANT_API_KEY"]
    missing = [k for k in critical if not os.environ.get(k, "").strip()]
    llm_ready = bool(
        (os.environ.get("RAG_LLM_BASE_URL", "").strip() and (
            os.environ.get("RAG_LLM_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
        ))
        or os.environ.get("HF_API_TOKEN", "").strip()
    )
    if not llm_ready:
        missing.append("RAG_LLM_BASE_URL+RAG_LLM_API_KEY (or HF_API_TOKEN)")

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
        try:
            ctx = resolve_request_context(
                user_profile=request.user_profile,
                chat_history=request.chat_history,
                conversation_history=request.conversation_history,
                legacy_stakeholder_type=request.stakeholder_type,
                legacy_audience_instructions=request.audience_instructions,
                session_id=request.session_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        logger.info(
            "query request received (len=%d, has_history=%s, stakeholder=%s)",
            len(request.query or ""),
            ctx.has_client_history,
            bool(ctx.stakeholder_type),
        )

        from ml.rag.graph import run_rag

        session_id, prior_summary, prior_recent = _resolve_prior_memory(
            request.session_id,
            ctx.history_messages,
        )

        create_trace(
            "rag.query",
            session_id=session_id,
            stakeholder_type=ctx.stakeholder_type,
            query=request.query,
            has_history=ctx.has_client_history,
        )
        if request.geo_override and str(request.geo_override).strip():
            logger.debug(
                "geo_override is deprecated and ignored; use user_profile.country for farmers_communities"
            )

        kwargs: dict = {}
        if prior_summary.strip() or prior_recent:
            kwargs["conversation_summary"] = prior_summary
            kwargs["recent_turns"] = prior_recent
        if ctx.user_profile is not None:
            kwargs["user_profile"] = ctx.user_profile
        if ctx.stakeholder_type:
            kwargs["stakeholder_type"] = ctx.stakeholder_type
        if ctx.audience_instructions:
            kwargs["audience_instructions"] = ctx.audience_instructions
        for key in (
            "time_start_override",
            "time_end_override",
            "news_top_k",
            "academic_top_k",
            "bq_top_k",
            "rerank_top_k",
            "ota_top_k",
        ):
            val = getattr(request, key, None)
            if val is not None:
                if isinstance(val, str):
                    val = val.strip()
                if val:
                    kwargs[key] = val

        result = run_rag(request.query, **kwargs)
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
        if not ctx.has_client_history:
            _persist_session_turn(session_id, request.query.strip(), answer)

        raw_citations = result.get("citations") or []
        citations = [CitationItem.model_validate(c) for c in raw_citations if isinstance(c, dict)]
        usage = UsageStats.from_usage_dict(result.get("usage") if isinstance(result.get("usage"), dict) else None)

        return QueryResponse(
            answer=answer,
            citations=citations,
            session_id=session_id,
            usage=usage,
            error=result.get("error"),
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
