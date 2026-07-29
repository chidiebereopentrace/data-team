"""
Public (exposition) chatbot API — versioned routes only; no retrieval internals.

Run: uvicorn ml.serving.chat.app:app --host 0.0.0.0 --port 7861
"""
from __future__ import annotations

import os
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
_env = _repo_root / "data" / "local" / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml.rag.chat_turn import create_session, execute_chat_turn
from ml.rag.chatbot.plan_policy import PLAN_ROUTE_SLUGS, PLAN_TYPES, default_category_for_plan
from ml.rag.chatbot.stakeholder_prompts import CATEGORIES
from ml.rag.api_schemas import CitationItem, UsageStats
from ml.rag.observability import flush_langfuse
from ml.rag.request_context import bootstrap_category, resolve_request_context
from ml.serving.chat.schemas import (
    ChatRequest,
    ChatSuccessResponse,
    SessionCreateRequest,
    SessionCreateResponse,
)

@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    yield
    flush_langfuse()


app = FastAPI(
    title="OpenTrace Chatbot API",
    description="Public v1 API for the OpenTrace chatbot (sessions, plan-aware answers).",
    version="1.0.0",
    lifespan=_app_lifespan,
)

_cors = os.environ.get("CHATBOT_CORS_ORIGINS", "*").strip().split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/v1")


@router.get("/health")
async def v1_health():
    return {"status": "ok", "service": "chatbot"}


@router.get("/meta")
async def v1_meta():
    return {
        "api_version": "1.0",
        "schema_version": "2",
        "build": os.environ.get("CHATBOT_BUILD_ID", "").strip() or None,
        "plan_types": list(PLAN_TYPES),
        "categories": list(CATEGORIES),
        "plan_routes": {slug: f"/v1/chat/{slug}" for slug in PLAN_ROUTE_SLUGS},
    }


@router.post("/sessions", response_model=SessionCreateResponse)
async def v1_create_session(body: SessionCreateRequest):
    try:
        sid = create_session(body.category)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return SessionCreateResponse(
        session_id=sid,
        created_at=datetime.now(timezone.utc).isoformat(),
        category=body.category,
    )


@router.post("/sessions/{plan_type_slug}", response_model=SessionCreateResponse)
async def v1_create_plan_session(plan_type_slug: str):
    """Create a session scoped to a specific plan type.

    The plan_type is inferred from the URL slug (e.g. /v1/sessions/farmers).
    Category defaults to the natural category for the plan; can be overridden
    by passing ?category=... as a query parameter in future.
    """
    slug = plan_type_slug.strip().lower()
    plan_type = PLAN_ROUTE_SLUGS.get(slug)
    if plan_type is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown plan slug {slug!r}. Valid slugs: {sorted(PLAN_ROUTE_SLUGS)}",
        )
    default_cat = default_category_for_plan(plan_type)
    # Free and Integrated have no default category — use the first available as a neutral fallback.
    # The session category only labels memory; it does not restrict what the user can ask.
    if default_cat is None:
        default_cat = CATEGORIES[0]["id"] if CATEGORIES else "Government"
    try:
        sid = create_session(default_cat)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return SessionCreateResponse(
        session_id=sid,
        created_at=datetime.now(timezone.utc).isoformat(),
        category=default_cat,
    )


async def _plan_chat(plan_type: str, body: ChatRequest, request_id: str):
    """Shared logic for plan-scoped chat routes.

    plan_type is injected from the route — the payload cannot override it.
    Auth/subscription enforcement is handled upstream by the API gateway.
    """
    try:
        ctx = resolve_request_context(
            user_profile=body.user_profile,
            chat_history=body.chat_history,
            conversation_history=body.conversation_history,
            session_id=body.session_id,
            injected_plan_type=plan_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    hist = ctx.history_messages
    if hist is not None and len(hist) == 0:
        raise HTTPException(status_code=422, detail="chat_history, if sent, must be non-empty")

    bootstrap_cat = bootstrap_category(body.user_profile) or default_category_for_plan(plan_type)
    sid = (body.session_id or "").strip() or None

    try:
        if hist is None:
            if not sid:
                if bootstrap_cat is None:
                    bootstrap_cat = CATEGORIES[0]["id"] if CATEGORIES else "Government"
                sid = create_session(bootstrap_cat)
        elif not sid and bootstrap_cat is not None:
            sid = create_session(bootstrap_cat)

        turn = execute_chat_turn(
            body.user_text(),
            session_id=sid,
            user_id=body.user_id,
            chat_history=hist,
            plan_type=ctx.plan_type,
            category=ctx.category,
            user_profile=ctx.user_profile,
            persist_to_session=(hist is None),
        )

        if turn.pipeline_error:
            payload = {
                "error": {"code": "rag_pipeline_error", "message": turn.pipeline_error},
                "session_id": turn.session_id,
                "plan_type": plan_type,
            }
            if os.environ.get("CHATBOT_DEBUG", "").strip().lower() in ("1", "true", "on"):
                payload["debug"] = {"request_id": request_id}
            return JSONResponse(status_code=502, content=payload)

        created_at = datetime.now(timezone.utc).isoformat()
        raw_citations = turn.citations or []
        citations = [CitationItem.model_validate(c) for c in raw_citations if isinstance(c, dict)]
        usage = UsageStats.from_usage_dict(turn.usage)
        return ChatSuccessResponse(
            assistant_message=turn.answer,
            citations=citations,
            session_id=turn.session_id,
            usage=usage,
            request_id=request_id,
            created_at=created_at,
            plan_type=plan_type,
            langfuse_trace_id=turn.langfuse_trace_id,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        detail = str(e)
        if os.environ.get("CHATBOT_DEBUG", "").strip().lower() in ("1", "true", "on"):
            detail += "\n\n" + traceback.format_exc()
        raise HTTPException(status_code=500, detail=detail) from e


# ---------------------------------------------------------------------------
# Plan-scoped chat routes (ML-034)
# The plan tier is locked by the URL — the payload cannot override it.
# ---------------------------------------------------------------------------

@router.post("/chat/free")
async def v1_chat_free(body: ChatRequest):
    """POST /v1/chat/free — Free tier (single country, top-line answers only)."""
    return await _plan_chat("Free", body, uuid.uuid4().hex)


@router.post("/chat/farmers")
async def v1_chat_farmers(body: ChatRequest):
    """POST /v1/chat/farmers — Farmers tier (localized crop/rainfall/market)."""
    return await _plan_chat("Farmers", body, uuid.uuid4().hex)


@router.post("/chat/government")
async def v1_chat_government(body: ChatRequest):
    """POST /v1/chat/government — Government tier (national/sub-national, food security)."""
    return await _plan_chat("Government", body, uuid.uuid4().hex)


@router.post("/chat/ngos")
async def v1_chat_ngos(body: ChatRequest):
    """POST /v1/chat/ngos — NGOs tier (multi-region risk, program angles)."""
    return await _plan_chat("NGOs", body, uuid.uuid4().hex)


@router.post("/chat/agribusinesses")
async def v1_chat_agribusinesses(body: ChatRequest):
    """POST /v1/chat/agribusinesses — Agribusinesses tier (cross-country, market volatility)."""
    return await _plan_chat("Agribusinesses", body, uuid.uuid4().hex)


@router.post("/chat/integrated")
async def v1_chat_integrated(body: ChatRequest):
    """POST /v1/chat/integrated — Integrated tier (full access, category lens per message)."""
    return await _plan_chat("Integrated", body, uuid.uuid4().hex)


@router.post("/chat")
async def v1_chat(body: ChatRequest):
    """POST /v1/chat — Generic chat endpoint (backward compatible).

    Prefer the plan-scoped routes (/v1/chat/{plan}) for new integrations.
    plan_type is read from user_profile.plan_type when using this endpoint.
    """
    request_id = uuid.uuid4().hex

    try:
        ctx = resolve_request_context(
            user_profile=body.user_profile,
            chat_history=body.chat_history,
            conversation_history=body.conversation_history,
            session_id=body.session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    hist = ctx.history_messages
    if hist is not None and len(hist) == 0:
        raise HTTPException(status_code=422, detail="chat_history, if sent, must be non-empty")

    bootstrap_cat = bootstrap_category(body.user_profile)
    sid = (body.session_id or "").strip() or None

    if hist is not None:
        if not sid and bootstrap_cat is None:
            raise HTTPException(
                status_code=422,
                detail="chat_history requires session_id or user_profile.category",
            )

    try:
        if hist is None:
            if not sid:
                if bootstrap_cat is None:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Send session_id from POST /v1/sessions, or omit session_id and send "
                            "user_profile with plan_type and category to bootstrap"
                        ),
                    )
                sid = create_session(bootstrap_cat)
        elif not sid and bootstrap_cat is not None:
            sid = create_session(bootstrap_cat)

        turn = execute_chat_turn(
            body.user_text(),
            session_id=sid,
            user_id=body.user_id,
            chat_history=hist,
            plan_type=ctx.plan_type,
            category=ctx.category,
            user_profile=ctx.user_profile,
            persist_to_session=(hist is None),
        )

        if turn.pipeline_error:
            payload = {
                "error": {
                    "code": "rag_pipeline_error",
                    "message": turn.pipeline_error,
                },
                "session_id": turn.session_id,
            }
            if os.environ.get("CHATBOT_DEBUG", "").strip().lower() in ("1", "true", "on"):
                payload["debug"] = {"request_id": request_id}
            return JSONResponse(status_code=502, content=payload)

        created_at = datetime.now(timezone.utc).isoformat()
        raw_citations = turn.citations or []
        citations = [CitationItem.model_validate(c) for c in raw_citations if isinstance(c, dict)]
        usage = UsageStats.from_usage_dict(turn.usage)
        return ChatSuccessResponse(
            assistant_message=turn.answer,
            citations=citations,
            session_id=turn.session_id,
            usage=usage,
            request_id=request_id,
            created_at=created_at,
            plan_type=ctx.plan_type,
            langfuse_trace_id=turn.langfuse_trace_id,
            plan_type=ctx.plan_type,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        detail = str(e)
        if os.environ.get("CHATBOT_DEBUG", "").strip().lower() in ("1", "true", "on"):
            detail += "\n\n" + traceback.format_exc()
        raise HTTPException(status_code=500, detail=detail) from e


app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "OpenTrace Chatbot API",
        "docs": "/docs",
        "health": "/v1/health",
        "meta": "/v1/meta",
        "sessions": "POST /v1/sessions",
        "sessions_plan": "POST /v1/sessions/{plan_type_slug}",
        "chat": "POST /v1/chat",
        "chat_plan": {slug: f"POST /v1/chat/{slug}" for slug in PLAN_ROUTE_SLUGS},
    }

