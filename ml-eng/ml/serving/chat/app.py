"""
Public (exposition) chatbot API — versioned routes only; no retrieval internals.

Run: uvicorn ml.serving.chat.app:app --host 0.0.0.0 --port 7861
"""
from __future__ import annotations

import os
import traceback
import uuid
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
from ml.rag.chatbot.stakeholder_prompts import STAKEHOLDER_TYPES
from ml.rag.api_schemas import CitationItem, UsageStats
from ml.rag.request_context import bootstrap_stakeholder_type, resolve_request_context
from ml.serving.chat.schemas import (
    ChatRequest,
    ChatSuccessResponse,
    SessionCreateRequest,
    SessionCreateResponse,
)

app = FastAPI(
    title="OpenTrace Chatbot API",
    description="Public v1 API for the OpenTrace chatbot (sessions, stakeholder-aware answers).",
    version="1.0.0",
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
        "schema_version": "1",
        "build": os.environ.get("CHATBOT_BUILD_ID", "").strip() or None,
        "stakeholder_types": list(STAKEHOLDER_TYPES),
    }


@router.post("/sessions", response_model=SessionCreateResponse)
async def v1_create_session(body: SessionCreateRequest):
    try:
        sid = create_session(body.stakeholder_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return SessionCreateResponse(
        session_id=sid,
        created_at=datetime.now(timezone.utc).isoformat(),
        stakeholder_type=body.stakeholder_type,
    )


@router.post("/chat")
async def v1_chat(body: ChatRequest):
    request_id = uuid.uuid4().hex

    try:
        ctx = resolve_request_context(
            user_profile=body.user_profile,
            chat_history=body.chat_history,
            conversation_history=body.conversation_history,
            legacy_stakeholder_type=body.stakeholder_type,
            session_id=body.session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    hist = ctx.history_messages
    if hist is not None and len(hist) == 0:
        raise HTTPException(status_code=422, detail="chat_history, if sent, must be non-empty")

    bootstrap_st = bootstrap_stakeholder_type(body.user_profile, body.stakeholder_type)
    sid = (body.session_id or "").strip() or None

    if hist is not None:
        if not sid and bootstrap_st is None:
            raise HTTPException(
                status_code=422,
                detail="chat_history requires session_id or user_profile.stakeholder_type",
            )

    try:
        if hist is None:
            if not sid:
                if bootstrap_st is None:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Send session_id from POST /v1/sessions, or omit session_id and send "
                            "user_profile.stakeholder_type to bootstrap"
                        ),
                    )
                sid = create_session(bootstrap_st)
        elif not sid and bootstrap_st is not None:
            sid = create_session(bootstrap_st)

        turn = execute_chat_turn(
            body.user_text(),
            session_id=sid,
            chat_history=hist,
            stakeholder_type=ctx.stakeholder_type,
            audience_instructions=ctx.audience_instructions,
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
        "chat": "POST /v1/chat",
    }
