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

import hashlib
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Use the single, authoritative environment loader for the entire RAG stack.
# This ensures config/.env (and RAG_* vars) are respected, and the service works
# in pure 12-factor mode when only env vars are supplied (the GCE production path).
from ml.rag.local_env import load_rag_dotenv

# ml-eng/ directory is parents[3] from ml/rag/app/api.py
_ml_eng = Path(__file__).resolve().parents[3]
load_rag_dotenv(_ml_eng)

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ml.rag.acf_signal import acf_signal_from_result
from ml.rag.api_schemas import (
    ACFSignal,
    ArtifactItem,
    CitationItem,
    UsageStats,
    UserProfile,
)
from ml.rag.chat_history import normalize_messages
from ml.rag.chat_memory import flat_messages_to_memory
from ml.rag.chat_turn import empty_session_blob, load_session, persist_session_turn
from ml.rag.chatbot.artifact_storage import refresh_artifact_url
from ml.rag.chatbot.bq_sql_diagnostics import (
    MAX_DIAGNOSTIC_SQL_STATEMENTS,
    sql_debug_rows_for_diagnostics,
    sql_statements_for_diagnostics,
)
from ml.rag.chatbot.capability_registry import registry_version
from ml.rag.chatbot.compile_sql_from_bind import RESULT_ROW_BUDGET
from ml.rag.chatbot.coverage_ledger import load_ledger
from ml.rag.chatbot.mart_indicator_classes import taxonomy_version
from ml.rag.chatbot.plan_policy import PLAN_ROUTE_SLUGS, allows_export
from ml.rag.chatbot.sql_compiler import sql_compiler_enabled
from ml.rag.chatbot.time_policy import trend_years
from ml.rag.chatbot.time_resolver import utc_today
from ml.rag.llm_model_config import resolved_llm_models
from ml.rag.observability import (
    flush_langfuse,
    get_current_trace_id,
    rag_trace_context,
    record_trace_score,
    tracing_release,
)
from ml.rag.rate_limiter import check_plan_rate_limit
from ml.rag.request_context import resolve_request_context
from ml.rag.retrievers.web_retriever import web_fallback_enabled
from ml.rag.session_store import (
    delete_session,
    get_idempotent_response,
    redis_status,
    save_idempotent_response,
    session_ttl_seconds,
)
from ml.serving.enterprise.auth import EnterpriseAuthMiddleware, get_request_tenant
from ml.serving.enterprise.metering import record_usage

logger = logging.getLogger("ml.rag.api")

_TRACE_BQ_SQL_DEBUG_MAX = MAX_DIAGNOSTIC_SQL_STATEMENTS


def _slim_bq_sql_plan_for_trace(plan: Any) -> dict[str, Any] | None:
    if not isinstance(plan, dict) or not plan:
        return None
    return {
        "selected_tables": plan.get("selected_tables"),
        "query_intents": plan.get("query_intents"),
        "skip_bq": plan.get("skip_bq"),
        "rationale": plan.get("rationale"),
        "slot_path": plan.get("slot_path"),
        "reasoner_job": plan.get("reasoner_job"),
    }


def _trace_bq_sql_debug_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return sql_debug_rows_for_diagnostics(result.get("bq_sql_debug"))


def _trace_submitted_bq_sql(result: dict[str, Any]) -> list[str]:
    rows = result.get("bq_sql_debug")
    if not isinstance(rows, list):
        return []
    return sql_statements_for_diagnostics(
        [
            row.get("sql")
            for row in rows
            if isinstance(row, dict) and row.get("job_id") and row.get("sql")
        ]
    )


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    yield
    flush_langfuse()


app = FastAPI(
    title="OpenTrace RAG API",
    description="Query BigQuery + vector DB via a graph RAG; use from the frontend chatbot.",
    version="0.1.0",
    lifespan=_app_lifespan,
)

app.add_middleware(EnterpriseAuthMiddleware)

# Added last so it wraps auth: preflights and 401/403 responses still carry CORS headers.
# No origins are allowed unless RAG_CORS_ORIGINS lists them (comma-separated).
_cors_origins = os.environ.get("RAG_CORS_ORIGINS", "").strip().split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "Idempotency-Key"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Natural language question for the RAG")
    include_trace: bool = Field(
        False,
        description=(
            "Include sanitized SQL, decomposition, retrieval diagnostics, and stage timings "
            "in response"
        ),
    )
    session_id: str | None = Field(
        None,
        description="Omit to start a new session; reuse for multi-turn chat (server-side memory)",
    )
    user_id: str | None = Field(
        None,
        description="Optional product user id for Langfuse user analytics (client-supplied until auth).",
    )
    chat_history: list[ChatMessage] | None = Field(
        None,
        description="Prior turns for this request (canonical). Server session is not updated when set.",
    )
    conversation_history: list[ChatMessage] | None = Field(
        None,
        description="Deprecated alias for chat_history.",
    )
    user_profile: UserProfile | None = Field(
        None,
        description="User profile: country, plan_type (access/geo), category (generation persona).",
    )
    time_start_override: str | None = None
    time_end_override: str | None = None
    news_top_k: int | None = None
    academic_top_k: int | None = None
    bq_top_k: int | None = None
    rerank_top_k: int | None = None
    ota_top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    acf: ACFSignal = Field(
        ...,
        description=(
            "ADZA Confidence Framework (Path B): band, score 0–100, and explanation "
            "from cited evidence. Surfaced on every response."
        ),
    )
    source_basis: str | None = Field(
        default=None,
        description=(
            "Ontology routing basis when ADZA_ONTOLOGY_ROUTING=1: structured, "
            "documents_only, web_only, or none. Omitted / null when routing is off."
        ),
    )
    session_id: str = Field(..., description="Pass on the next request for chat continuity")
    session_found: bool = Field(
        False,
        description=(
            "True when a prior server-side session blob was loaded for this request "
            "(Redis / in-memory). False for new sessions, expired/missing ids, or "
            "when chat_history is supplied (server session is not read)."
        ),
    )
    session_ttl_seconds: int = Field(
        default_factory=session_ttl_seconds,
        description="Configured server session TTL (RAG_SESSION_TTL_SECONDS, default 604800).",
    )
    usage: UsageStats = Field(default_factory=lambda: UsageStats())
    error: str | None = None
    trace: dict | None = None
    langfuse_trace_id: str | None = Field(
        None,
        description="Langfuse trace id when tracing is enabled (for feedback / debugging)",
    )
    artifacts: list[ArtifactItem] = Field(
        default_factory=list,
        description=(
            "Downloadable exports (CSV/chart/DOCX/PDF). Populated on Agribusinesses and "
            "Integrated plans when the query requests an export and builders succeed; "
            "otherwise empty. Signed URL TTL defaults to 86400s — refresh via "
            "GET /artifacts/{artifact_id}/url."
        ),
    )


class ArtifactUrlResponse(BaseModel):
    id: str
    filename: str
    url: str
    expires_in_seconds: int
    storage_uri: str | None = None


class TraceFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0, description="1.0 = thumbs up, 0.0 = thumbs down")
    comment: str | None = Field(None, max_length=500)


def _gcp_credentials_status() -> dict[str, Any]:
    """Validate ADC path JSON when present; report BASE64 presence for ops."""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    base64_set = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_BASE64", "").strip())
    info: dict[str, Any] = {
        "credentials_path_set": bool(path),
        "credentials_base64_set": base64_set,
        "path": path or None,
        "json_ok": False,
    }
    if not path:
        info["error"] = "GOOGLE_APPLICATION_CREDENTIALS not set"
        return info
    p = Path(path)
    if not p.is_file():
        info["error"] = f"credentials file missing: {path}"
        return info
    try:
        with p.open(encoding="utf-8") as fh:
            json.load(fh)
        info["json_ok"] = True
    except Exception as exc:
        info["error"] = f"credentials JSON invalid: {exc}"
    return info


def _bq_readiness() -> dict[str, Any]:
    """
    Nested BigQuery readiness. When BQ_PROJECT is set, credentials must parse and a
    lightweight datasets.list must succeed — otherwise the API is not_ready.
    """
    project = os.environ.get("BQ_PROJECT", "").strip()
    gcp = _gcp_credentials_status()
    info: dict[str, Any] = {
        "project_set": bool(project),
        "project": project or None,
        "gcp": gcp,
        "ok": False,
    }
    if not project:
        info["ok"] = True
        info["skipped"] = "BQ_PROJECT unset"
        return info
    if not gcp.get("json_ok"):
        info["error"] = gcp.get("error") or "GCP credentials not ready"
        return info
    try:
        from google.cloud import bigquery  # lazy import

        client = bigquery.Client(project=project)
        # Soft connectivity probe — list at most one dataset.
        next(iter(client.list_datasets(max_results=1)), None)
        info["ok"] = True
    except Exception as exc:
        info["error"] = str(exc)
    return info


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
    When BQ_PROJECT is set, GCP credentials must be valid JSON and BigQuery must respond;
    otherwise status is not_ready (Free/paid NL2SQL depends on BQ).
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

    bq_info = _bq_readiness()
    bq_blocks = bool(bq_info.get("project_set")) and not bool(bq_info.get("ok"))
    if bq_blocks and "BQ_PROJECT+GCP credentials" not in missing:
        missing.append("BQ_PROJECT+GCP credentials")

    payload: dict[str, Any] = {
        "status": "ready" if not missing and not bq_blocks else "not_ready",
        "service": "rag",
        "missing_config_keys": missing,
        "bq": bq_info,
    }
    if redis_info:
        payload["redis"] = redis_info
    payload["llm_models"] = resolved_llm_models()
    return payload


def _version_settings() -> dict[str, Any]:
    """Behaviour-shaping settings only: flags, budgets and model ids. Never keys or URLs."""
    return {
        "sql_compiler": sql_compiler_enabled(),
        "slot_reasoner": os.environ.get("RAG_SLOT_REASONER", "").strip().lower() in ("1", "true", "yes", "on"),
        "web_fallback": web_fallback_enabled(),
        "trend_latest_years": trend_years(),
        "result_row_budget": RESULT_ROW_BUDGET,
        "llm_models": resolved_llm_models(),
    }


@app.get("/version")
async def version():
    """Build and contract identity: SHA, execution mode, catalog and ledger versions, settings fingerprint."""
    settings = _version_settings()
    ledger = load_ledger()
    return {
        "service": "rag",
        "sha": tracing_release() or "unknown",
        "execution_mode": "bind_compiler" if settings["sql_compiler"] else "nl2sql",
        "catalog_version": taxonomy_version(),
        "registry_version": registry_version(),
        "ledger_snapshot": ledger.get("snapshot_hash"),
        "ledger_built_at": ledger.get("built_at"),
        "coverage_built_at": ledger.get("coverage_built_at"),
        "settings_fingerprint": hashlib.sha256(
            json.dumps(settings, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    }


def _session_owner(http_request: Request) -> str:
    """Sessions belong to the authenticated tenant; with auth off there is one local owner."""
    tenant = get_request_tenant(http_request)
    return f"tenant:{tenant.tenant_id}" if tenant is not None else "local"


def _client_history_memory(
    history_messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """(conversation_summary, recent_turns) folded from client-supplied history."""
    return flat_messages_to_memory(normalize_messages(history_messages))


_GENERIC_500_DETAIL = "The question could not be answered due to an internal error; retry later."


def _augment_http_500_detail(detail: str, *, debug: bool) -> str:
    """Shape a 500 body. Exception text reaches clients only when RAG_DEBUG is on."""
    if debug:
        return detail + "\n\n" + traceback.format_exc()
    if "unsourced_figures" in detail:
        return "Answer grounding failed unexpectedly; retry the question."
    return _GENERIC_500_DETAIL


def _plan_slug_for(plan_type: str | None) -> str:
    for slug, name in PLAN_ROUTE_SLUGS.items():
        if name == plan_type:
            return slug
    return "free"


async def _run_query(
    request: QueryRequest,
    http_request: Request,
    *,
    injected_plan_type: str | None = None,
) -> QueryResponse:
    """Shared query logic. When injected_plan_type is set, path owns the plan tier.

    Without a plan route, an authenticated tenant's own plan applies.
    """
    tenant = get_request_tenant(http_request)
    if injected_plan_type is None and tenant is not None:
        injected_plan_type = PLAN_ROUTE_SLUGS[tenant.plan_slug]
    owner = _session_owner(http_request)
    idempotency_key = http_request.headers.get("idempotency-key", "").strip()
    if idempotency_key:
        replay = get_idempotent_response(owner, idempotency_key)
        if replay is not None:
            return QueryResponse.model_validate(replay)
    langfuse_trace_id: str | None = None
    try:
        client_history = request.chat_history is not None or request.conversation_history is not None
        session_id = (request.session_id or "").strip() or uuid.uuid4().hex
        if client_history:
            loaded, session_found = empty_session_blob(), False
        else:
            loaded, session_found = load_session(owner, session_id)
        try:
            ctx = resolve_request_context(
                user_profile=request.user_profile,
                chat_history=request.chat_history,
                conversation_history=request.conversation_history,
                session_blob=None if client_history else loaded,
                injected_plan_type=injected_plan_type,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        check_plan_rate_limit(_plan_slug_for(ctx.plan_type), http_request)

        logger.info(
            "query request received (len=%d, has_history=%s, plan_type=%s, category=%s)",
            len(request.query or ""),
            ctx.has_client_history,
            ctx.plan_type,
            ctx.category,
        )

        from ml.rag.graph import run_rag

        if ctx.history_messages is not None:
            prior_summary, prior_recent = _client_history_memory(ctx.history_messages)
        else:
            prior_summary = str(loaded.get("conversation_summary") or "")
            prior_recent = normalize_messages(loaded.get("recent_turns"))

        kwargs: dict = {}
        if prior_summary.strip() or prior_recent:
            kwargs["conversation_summary"] = prior_summary
            kwargs["recent_turns"] = prior_recent
        if isinstance(loaded.get("active_request"), dict):
            kwargs["active_request"] = loaded["active_request"]
        if isinstance(loaded.get("last_bq_facts"), list):
            kwargs["last_bq_facts"] = loaded["last_bq_facts"]
        if ctx.user_profile is not None:
            kwargs["user_profile"] = ctx.user_profile
        if ctx.plan_type:
            kwargs["plan_type"] = ctx.plan_type
        if ctx.category:
            kwargs["category"] = ctx.category
        kwargs["export_enabled"] = allows_export(ctx.plan_type)
        kwargs["session_id"] = session_id
        kwargs["trace_tags"] = ["api"]
        kwargs["reference_date"] = utc_today().isoformat()
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

        with rag_trace_context(
            trace_name="rag.query",
            session_id=session_id,
            user_id=request.user_id,
            plan_type=ctx.plan_type,
            category=ctx.category,
            trace_input={"query": request.query[:500]},
            tags=["api"],
        ) as trace_handle:
            langfuse_trace_id = get_current_trace_id()
            rag_started_at = time.perf_counter()
            result = run_rag(request.query, **kwargs)
            rag_pipeline_ms = (time.perf_counter() - rag_started_at) * 1000
            trace_handle.update_output(result, latency_ms=rag_pipeline_ms)
            langfuse_trace_id = get_current_trace_id() or langfuse_trace_id
        flush_langfuse()
        trace: dict | None = None
        trace_allowed = tenant.allow_trace if tenant is not None else True
        if request.include_trace and trace_allowed:
            trace = {
                "decomposition": result.get("decomposition"),
                "bq_table_candidates_count": len(result.get("bq_table_candidates") or []),
                "vector_news_count": len(result.get("vector_news_results") or []),
                "vector_academic_papers_count": len(result.get("vector_academic_papers_results") or []),
                "vector_policies_count": len(result.get("vector_policies_results") or []),
                "vector_public_reports_count": len(result.get("vector_public_reports_results") or []),
                "vector_formation_count": len(result.get("vector_formation_results") or []),
                "vector_academic_count": len(result.get("vector_academic_results") or []),
                "vector_ota_count": len(result.get("vector_ota_results") or []),
                "merged_context_count": len(result.get("merged_context") or []),
                "reranked_context_count": len(result.get("reranked_context") or []),
                "langfuse_trace_id": langfuse_trace_id,
                "rag_pipeline_ms": rag_pipeline_ms,
                "bq_sql_queries": sql_statements_for_diagnostics(result.get("bq_sql_queries")),
                "bq_planned_sql": sql_statements_for_diagnostics(
                    (result.get("bq_sql_plan") or {}).get("bq_sql_queries")
                    if isinstance(result.get("bq_sql_plan"), dict)
                    else None
                ),
                "bq_submitted_sql": _trace_submitted_bq_sql(result),
                "bq_sql_debug": _trace_bq_sql_debug_rows(result),
                "bq_sql_plan": _slim_bq_sql_plan_for_trace(result.get("bq_sql_plan")),
                "sql_source": result.get("sql_source"),
                "bq_cache_hit": result.get("bq_cache_hit"),
                "bq_nl2sql_ms": result.get("bq_nl2sql_ms"),
                "bq_execute_ms": result.get("bq_execute_ms"),
                "decompose_llm_ms": result.get("decompose_llm_ms"),
                "vector_ms": result.get("vector_ms"),
                "rerank_ms": result.get("rerank_ms"),
                "generate_ms": result.get("generate_ms"),
                "supervisor_plan": result.get("supervisor_plan"),
                "value_hits": result.get("value_hits")
                or (result.get("bq_sql_plan") or {}).get("value_hits"),
                "user_query": result.get("user_query") or request.query,
                "context_rewrite": result.get("context_rewrite"),
                "detail_rewrite": result.get("detail_rewrite"),
                "context_applied": result.get("context_applied"),
                "user_query_dropped": result.get("user_query_dropped"),
                "vector_cache_hit": result.get("vector_cache_hit"),
                "coverage_retry": result.get("coverage_retry"),
                "vector_texts_used": result.get("vector_texts_used"),
            }

        answer = result.get("answer", "") or ""
        if not ctx.has_client_history:
            profile = ctx.user_profile or {}
            saved = persist_session_turn(
                owner,
                session_id,
                request.query.strip(),
                answer,
                loaded=loaded,
                result=result,
                category=ctx.category,
                plan_type=ctx.plan_type,
                country=str(profile.get("country") or "").strip() or None,
            )
            if not saved:
                logger.warning("session %s moved on during this turn; stale state not written", session_id)

        raw_citations = result.get("citations") or []
        citations = [CitationItem.model_validate(c) for c in raw_citations if isinstance(c, dict)]
        raw_artifacts = result.get("artifacts") or []
        artifacts = [ArtifactItem.model_validate(a) for a in raw_artifacts if isinstance(a, dict)]
        usage = UsageStats.from_usage_dict(result.get("usage") if isinstance(result.get("usage"), dict) else None)
        if tenant is not None:
            record_usage(
                tenant,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                export_count=len(artifacts),
            )

        acf = acf_signal_from_result(result)

        source_basis = result.get("source_basis")
        if source_basis is not None:
            source_basis = str(source_basis).strip() or None

        response = QueryResponse(
            answer=answer,
            citations=citations,
            acf=acf,
            source_basis=source_basis,
            session_id=session_id,
            session_found=session_found,
            session_ttl_seconds=session_ttl_seconds(),
            usage=usage,
            error=result.get("error"),
            trace=trace,
            langfuse_trace_id=langfuse_trace_id,
            artifacts=artifacts,
        )
        if idempotency_key:
            save_idempotent_response(owner, idempotency_key, response.model_dump(mode="json"))
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("query failed (langfuse_trace_id=%s)", langfuse_trace_id)
        debug = os.environ.get("RAG_DEBUG", "").strip().lower() in ("1", "true", "on")
        detail = _augment_http_500_detail(str(e), debug=debug)
        if langfuse_trace_id:
            detail = f"{detail} (trace id: {langfuse_trace_id})"
        raise HTTPException(status_code=500, detail=detail) from e


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request):
    """Run the RAG pipeline with the caller's tenant plan. Prefer /query/{plan} routes."""
    return await _run_query(request, http_request)


@app.post("/query/free", response_model=QueryResponse, summary="Query (Free plan)")
async def query_free(request: QueryRequest, http_request: Request):
    """Free tier — single country, top-line answers. Path locks plan_type; body cannot escalate."""
    return await _run_query(request, http_request, injected_plan_type="Free")


@app.post("/query/farmers", response_model=QueryResponse, summary="Query (Farmers plan)")
async def query_farmers(request: QueryRequest, http_request: Request):
    """Farmers tier — localized crop/rainfall/market. Path locks plan_type."""
    return await _run_query(request, http_request, injected_plan_type="Farmers")


@app.post("/query/government", response_model=QueryResponse, summary="Query (Government plan)")
async def query_government(request: QueryRequest, http_request: Request):
    """Government tier — national/sub-national food security. Path locks plan_type."""
    return await _run_query(request, http_request, injected_plan_type="Government")


@app.post("/query/ngos", response_model=QueryResponse, summary="Query (NGOs plan)")
async def query_ngos(request: QueryRequest, http_request: Request):
    """NGOs tier — multi-region risk and program angles. Path locks plan_type."""
    return await _run_query(request, http_request, injected_plan_type="NGOs")


@app.post("/query/agribusinesses", response_model=QueryResponse, summary="Query (Agribusinesses plan)")
async def query_agribusinesses(request: QueryRequest, http_request: Request):
    """Agribusinesses tier — cross-country, market volatility. Path locks plan_type."""
    return await _run_query(request, http_request, injected_plan_type="Agribusinesses")


@app.post("/query/integrated", response_model=QueryResponse, summary="Query (Integrated plan)")
async def query_integrated(request: QueryRequest, http_request: Request):
    """Integrated tier — full access; category lens per message. Path locks plan_type."""
    return await _run_query(request, http_request, injected_plan_type="Integrated")


@app.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str, http_request: Request):
    """
    Delete a session and its conversation memory.

    Sprint 1, Week 2: explicit session lifecycle management. The frontend should call
    this when the user starts a new conversation or logs out, ensuring no stale memory
    bleeds into future queries.
    """
    sid = session_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id must not be empty")
    delete_session(_session_owner(http_request), sid)
    logger.info("session deleted: %s", sid)
    return {"status": "deleted", "session_id": sid}


@app.get("/artifacts/{artifact_id}/url", response_model=ArtifactUrlResponse)
async def get_artifact_url(
    artifact_id: str,
    filename: str = Query(..., min_length=1, description="Original artifact filename from the query response"),
):
    """
    Re-sign a download URL for an existing export artifact.

    Signed URLs expire after ``RAG_ARTIFACT_SIGNED_URL_TTL_SECONDS`` (default **86400**).
    Pass the ``id`` and ``filename`` from the original ``artifacts[]`` item.
    """
    try:
        meta = refresh_artifact_url(artifact_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("artifact re-sign failed id=%s filename=%s", artifact_id, filename)
        raise HTTPException(status_code=500, detail="Failed to refresh artifact URL") from exc
    return ArtifactUrlResponse.model_validate(meta)


@app.post("/feedback")
async def trace_feedback(request: TraceFeedbackRequest):
    """Record user feedback (thumbs up/down) on a Langfuse trace."""
    ok = record_trace_score(
        trace_id=request.trace_id,
        name="user_feedback",
        value=request.score,
        comment=request.comment,
    )
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="Langfuse tracing is not configured or trace id is invalid",
        )
    return {"status": "ok", "trace_id": request.trace_id}


@app.get("/")
async def root():
    return {
        "message": "OpenTrace RAG API",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "query": "POST /query",
        "artifact_url": "GET /artifacts/{artifact_id}/url?filename=...",
        "plan_routes": {slug: f"POST /query/{slug}" for slug in PLAN_ROUTE_SLUGS},
    }
