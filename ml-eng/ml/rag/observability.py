"""
Optional Langfuse observability for the RAG pipeline (SDK v3+, OTEL-native).

Gracefully no-ops when LANGFUSE_* keys are absent (fail-open for HF/Railway deploys).
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

if TYPE_CHECKING:
    from contextvars import Token
    from langchain_core.runnables import RunnableConfig
else:
    Token = Any  # type: ignore[misc,assignment]

try:
    from langfuse import get_client, observe, propagate_attributes
except Exception:  # pragma: no cover - langfuse not installed
    get_client = None  # type: ignore[assignment]
    observe = None  # type: ignore[assignment]
    propagate_attributes = None  # type: ignore[assignment]

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except Exception:  # pragma: no cover - langchain package missing
    LangfuseCallbackHandler = None  # type: ignore[assignment]


_openrouter_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "openrouter_run_id", default=None
)


def openrouter_sessions_enabled() -> bool:
    """True when OpenRouter is the LLM backend and session bundling is not disabled."""
    base = os.environ.get("RAG_LLM_BASE_URL", "").strip().lower()
    if "openrouter.ai" not in base:
        return False
    flag = os.environ.get("RAG_OPENROUTER_SESSION_ID", "on").strip().lower()
    return flag not in ("0", "off", "false", "no")


def set_openrouter_run_id(run_id: str | None) -> Token:
    """Set the OpenRouter session id for the current async/task context."""
    return _openrouter_run_id.set(run_id)


def get_openrouter_run_id() -> str | None:
    """Return the active OpenRouter session id, if any."""
    return _openrouter_run_id.get()


def reset_openrouter_run_id(token: Token) -> None:
    """Restore the previous OpenRouter session id."""
    _openrouter_run_id.reset(token)


@contextmanager
def openrouter_run_context(run_id: str) -> Iterator[None]:
    """Scope all LLM calls under one OpenRouter session id for the duration of a RAG run."""
    rid = (run_id or "").strip()
    if not rid:
        yield
        return
    token = set_openrouter_run_id(rid[:256])
    try:
        yield
    finally:
        reset_openrouter_run_id(token)


def _langfuse_base_url() -> str | None:
    """Resolve Langfuse host (v3 prefers LANGFUSE_BASE_URL; LANGFUSE_HOST is legacy)."""
    for key in ("LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        val = os.environ.get(key, "").strip()
        if val:
            return val.rstrip("/")
    return None


def _ensure_langfuse_env() -> None:
    """Map legacy LANGFUSE_HOST to LANGFUSE_BASE_URL before client init."""
    if not os.environ.get("LANGFUSE_BASE_URL", "").strip():
        host = os.environ.get("LANGFUSE_HOST", "").strip()
        if host:
            os.environ["LANGFUSE_BASE_URL"] = host.rstrip("/")


def is_tracing_enabled() -> bool:
    if get_client is None:
        return False
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


@lru_cache(maxsize=1)
def get_langfuse_client() -> Any | None:
    """Return the global Langfuse client when keys are configured, else None."""
    if get_client is None or not is_tracing_enabled():
        return None
    _ensure_langfuse_env()
    try:
        return get_client()
    except Exception:
        return None


def get_langfuse() -> Any | None:
    """Back-compat alias for ``get_langfuse_client``."""
    return get_langfuse_client()


def get_langfuse_callback() -> Any | None:
    """LangChain/LangGraph callback handler for graph node tracing."""
    if LangfuseCallbackHandler is None or not is_tracing_enabled():
        return None
    try:
        return LangfuseCallbackHandler()
    except Exception:
        return None


def flush_langfuse() -> None:
    """Flush pending trace events (call on FastAPI shutdown or script exit)."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass


def _build_tags(
    *,
    plan_type: str | None = None,
    category: str | None = None,
    extra_tags: list[str] | None = None,
    route: str | None = None,
) -> list[str]:
    tags: list[str] = list(extra_tags or [])
    env = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", "").strip()
    if env:
        tags.append(f"env:{env}")
    release = os.environ.get("LANGFUSE_TRACING_RELEASE", "").strip()
    if release:
        tags.append(f"release:{release}")
    if route:
        tags.append(f"route:{route}")
    if plan_type:
        tags.append(f"plan_type:{plan_type}")
    if category:
        tags.append(f"category:{category}")
    return tags


def sql_hash(sql: str) -> str:
    """Short deterministic hash for SQL in trace metadata (not full SQL)."""
    normalized = (sql or "").strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def summarize_rag_result_for_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Retrieval/rerank counts for root trace output metadata."""
    route = infer_rag_route(result)
    summary: dict[str, Any] = {
        "route": route,
        "vector_news_count": len(result.get("vector_news_results") or []),
        "vector_academic_count": len(result.get("vector_academic_results") or []),
        "vector_ota_count": len(result.get("vector_ota_results") or []),
        "bq_table_candidates_count": len(result.get("bq_table_candidates") or []),
        "bq_results_count": len(result.get("bq_results") or []),
        "merged_context_count": len(result.get("merged_context") or []),
        "reranked_context_count": len(result.get("reranked_context") or []),
        "web_results_count": len(result.get("web_results") or []),
    }
    rerank_mode = result.get("rerank_mode") or result.get("_rerank_mode")
    if rerank_mode:
        summary["rerank_mode"] = str(rerank_mode)
    if result.get("error"):
        summary["error"] = str(result.get("error"))[:200]
    if result.get("latency_ms") is not None:
        summary["latency_ms"] = float(result["latency_ms"])
    return summary


def infer_rag_route(result: dict[str, Any]) -> str:
    """Derive pipeline route label from a ``run_rag()`` result dict."""
    if result.get("is_meta_query"):
        return "meta"
    if result.get("is_product_query"):
        return "product"
    if result.get("insufficient_context"):
        return "insufficient"
    if result.get("web_results"):
        return "full_rag + web_fallback"
    return "full_rag"


def build_rag_invoke_config(
    *,
    base_config: RunnableConfig | dict[str, Any] | None = None,
    session_id: str | None = None,
    plan_type: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
) -> RunnableConfig:
    """Build LangGraph ``RunnableConfig`` with Langfuse callback and metadata."""
    base: dict[str, Any] = dict(base_config) if base_config else {}
    metadata = dict(base.get("metadata") or {})
    if session_id:
        metadata["langfuse_session_id"] = session_id
    tag_list = _build_tags(plan_type=plan_type, category=category, extra_tags=tags)
    if tag_list:
        metadata["langfuse_tags"] = tag_list
    base["metadata"] = metadata
    handler = get_langfuse_callback()
    if handler:
        callbacks = list(base.get("callbacks") or [])
        if handler not in callbacks:
            callbacks.append(handler)
        base["callbacks"] = callbacks
    return cast("RunnableConfig", base)


@dataclass
class RagTraceHandle:
    """Handle for updating a root RAG trace after pipeline completion."""

    span: Any | None = None
    _closed: bool = field(default=False, repr=False)

    def update_output(
        self,
        result: dict[str, Any],
        *,
        route: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        if self.span is None or self._closed:
            return
        route_label = route or infer_rag_route(result)
        answer = str(result.get("answer") or "")
        err = result.get("error")
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        retrieval_summary = summarize_rag_result_for_trace(result)
        if latency_ms is not None:
            retrieval_summary["latency_ms"] = latency_ms
        output: dict[str, Any] = {
            "route": route_label,
            "answer_preview": answer[:500] if answer else "",
            "error": str(err).strip() if err else None,
            "usage": usage,
            **retrieval_summary,
        }
        tag_list = _build_tags(route=route_label)
        try:
            self.span.update_trace(
                output=output,
                metadata={"route": route_label, **retrieval_summary},
                tags=tag_list or None,
            )
        except Exception:
            try:
                self.span.update(output=output, metadata=retrieval_summary)
            except Exception:
                pass


@contextmanager
def rag_trace_context(
    *,
    trace_name: str = "rag.query",
    session_id: str | None = None,
    plan_type: str | None = None,
    category: str | None = None,
    trace_input: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[RagTraceHandle]:
    """
    Root span for one RAG request with session/plan/category propagated to children.

    Yields a :class:`RagTraceHandle` — call ``update_output(result)`` after ``run_rag()``.
    Also sets OpenRouter ``session_id`` (trace id) for LLM cost bundling when enabled.
    """
    handle = RagTraceHandle()
    client = get_langfuse_client()
    fallback_run_id = uuid.uuid4().hex

    if client is None or propagate_attributes is None:
        with openrouter_run_context(fallback_run_id):
            yield handle
        return

    tag_list = _build_tags(plan_type=plan_type, category=category, extra_tags=tags)
    propagate_kwargs: dict[str, Any] = {}
    if session_id:
        propagate_kwargs["session_id"] = session_id
    if tag_list:
        propagate_kwargs["tags"] = tag_list

    try:
        with client.start_as_current_observation(
            as_type="span",
            name=trace_name,
            input=trace_input,
        ) as root_span:
            handle.span = root_span
            run_id = get_current_trace_id() or fallback_run_id
            try:
                root_span.update_trace(metadata={"openrouter_session_id": run_id})
            except Exception:
                pass
            with openrouter_run_context(run_id):
                if propagate_kwargs:
                    with propagate_attributes(**propagate_kwargs):
                        yield handle
                else:
                    yield handle
            handle._closed = True
    except Exception:
        with openrouter_run_context(fallback_run_id):
            yield handle


def safe_llm_trace_input(messages: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Redacted LLM input for Langfuse (no full history or secrets)."""
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = str(msg.get("content") or "")[:500]
            break
    return {
        "model": model,
        "message_count": len(messages),
        "last_user_message": last_user,
    }


def update_current_llm_generation(
    *,
    input_data: dict[str, Any] | None = None,
    output: str | None = None,
    model: str | None = None,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Update the active ``@observe`` generation span; fail-open."""
    client = get_langfuse_client()
    if client is None:
        return
    kwargs: dict[str, Any] = {}
    if input_data is not None:
        kwargs["input"] = input_data
    if output is not None:
        kwargs["output"] = output[:500] if output else ""
    if model is not None:
        kwargs["model"] = model
    if metadata:
        kwargs["metadata"] = metadata
    if usage:
        kwargs["usage_details"] = {
            "input": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "output": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            "total": int(usage.get("total_tokens") or 0),
        }
    if not kwargs:
        return
    try:
        client.update_current_generation(**kwargs)
    except Exception:
        try:
            client.update_current_span(**kwargs)
        except Exception:
            pass


def get_observe_decorator() -> Any:
    """Return Langfuse ``@observe`` or a no-op decorator when tracing is disabled."""
    if observe is None or not is_tracing_enabled():
        def _noop(*_args: Any, **_kwargs: Any):
            def decorator(fn: Any) -> Any:
                return fn
            return decorator
        return _noop
    return observe


def update_current_span_metadata(metadata: dict[str, Any]) -> None:
    """Attach metadata to the active span; fail-open when tracing is off."""
    if not metadata:
        return
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.update_current_span(metadata=metadata)
    except Exception:
        pass


@contextmanager
def observed_span(
    name: str,
    *,
    input_data: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Context manager for a nested pipeline span (retrieval, rerank, embedding)."""
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
        ) as span:
            yield span
    except Exception:
        yield None


def run_with_tracing_context(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[[], Any]:
    """
    Capture OTEL + contextvar state from the calling thread.

    Returns a no-arg callable suitable for ``ThreadPoolExecutor.submit``.
    """
    var_ctx = contextvars.copy_context()
    otel_ctx: Any = None
    try:
        from opentelemetry import context as otel_context

        otel_ctx = otel_context.get_current()
    except Exception:
        pass

    def _bound() -> Any:
        def _call() -> Any:
            if otel_ctx is not None:
                try:
                    from opentelemetry import context as otel_context

                    token = otel_context.attach(otel_ctx)
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        otel_context.detach(token)
                except Exception:
                    return fn(*args, **kwargs)
            return fn(*args, **kwargs)

        return var_ctx.run(_call)

    return _bound


def get_current_trace_id() -> str | None:
    """Return the active Langfuse trace id when tracing is enabled."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:
        return None


def record_trace_score(
    *,
    name: str,
    value: float | int | bool,
    trace_id: str | None = None,
    comment: str | None = None,
) -> bool:
    """
    Record user feedback or eval score on a trace (Langfuse Scores API).

    Returns True when the score was submitted.
    """
    client = get_langfuse_client()
    if client is None:
        return False
    tid = (trace_id or "").strip() or get_current_trace_id()
    if not tid:
        return False
    try:
        client.create_score(
            trace_id=tid,
            name=name,
            value=float(value) if isinstance(value, bool) else value,
            comment=comment,
        )
        return True
    except Exception:
        return False


def trace_elapsed_ms(start: float) -> float:
    """Milliseconds since ``time.perf_counter()`` start."""
    return round((time.perf_counter() - start) * 1000.0, 2)


# Back-compat shim (prefer rag_trace_context)
def create_trace(name: str, **metadata: Any) -> Any | None:
    """Deprecated: use ``rag_trace_context`` for unified traces."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        span = client.start_span(name=name, metadata=metadata or None)
        return span
    except Exception:
        return None
