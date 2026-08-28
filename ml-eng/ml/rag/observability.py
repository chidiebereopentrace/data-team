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
def _langfuse_client_cached() -> Any | None:
    """Cached client init — only called when tracing keys are present."""
    _ensure_langfuse_env()
    try:
        return get_client()
    except Exception:
        return None


def get_langfuse_client() -> Any | None:
    """Return the global Langfuse client when keys are configured, else None.

    Disabled/missing-key results are not cached so keys loaded after import
    (dotenv / Railway inject) still activate tracing.
    """
    if get_client is None or not is_tracing_enabled():
        return None
    return _langfuse_client_cached()


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


def tracing_release() -> str:
    """Explicit LANGFUSE_TRACING_RELEASE, else Railway git SHA (short)."""
    explicit = os.environ.get("LANGFUSE_TRACING_RELEASE", "").strip()
    if explicit:
        return explicit
    sha = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "").strip()
    if sha:
        return sha[:12]
    return ""


def _build_tags(
    *,
    plan_type: str | None = None,
    category: str | None = None,
    extra_tags: list[str] | None = None,
    route: str | None = None,
    answer_lang: str | None = None,
    acf_band: str | None = None,
) -> list[str]:
    tags: list[str] = list(extra_tags or [])
    env = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", "").strip()
    if env:
        tags.append(f"env:{env}")
    release = tracing_release()
    if release:
        tags.append(f"release:{release}")
    if route:
        tags.append(f"route:{route}")
    if plan_type:
        tags.append(f"plan_type:{plan_type}")
    if category:
        tags.append(f"category:{category}")
    lang = (answer_lang or "").strip()
    if lang:
        tags.append(f"answer_lang:{lang}")
    band = (acf_band or "").strip()
    if band:
        tags.append(f"acf_band:{band}")
    return tags


def sql_hash(sql: str) -> str:
    """Short deterministic hash for SQL in trace metadata (not full SQL)."""
    normalized = (sql or "").strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _bq_soft_fail_flags(result: dict[str, Any]) -> dict[str, Any]:
    """Derive BQ soft-fail signals from result rows / metadata."""
    bq_results = result.get("bq_results") or []
    validation_failed = 0
    execution_failed = 0
    for row in bq_results:
        if not isinstance(row, dict):
            continue
        text = str(row.get("content") or row.get("text") or "")
        if "[BQ validation failed" in text:
            validation_failed += 1
        if "[BQ execution error" in text or "[BQ execution failed" in text:
            execution_failed += 1
    sql_count = result.get("bq_sql_count")
    if sql_count is None:
        sql_count = len(bq_results) if bq_results else 0
    return {
        "bq_sql_count": int(sql_count),
        "bq_validation_failed": validation_failed > 0,
        "bq_execution_failed": execution_failed > 0,
        "bq_validation_failed_count": validation_failed,
        "bq_execution_failed_count": execution_failed,
    }


def _acf_and_language_for_trace(result: dict[str, Any]) -> dict[str, Any]:
    """ACF Path B fields + soft answer_lang for root-trace metadata."""
    out: dict[str, Any] = {}
    for key in (
        "acf_band",
        "acf_band_label",
        "acf_score",
        "acf_claim_level",
        "acf_question_type",
        "acf_applied_ceiling",
        "acf_config_version",
    ):
        val = result.get(key)
        if val is not None and val != "":
            out[key] = val
    expl = result.get("acf_explanation") or result.get("acf_note")
    if expl is not None and str(expl).strip():
        out["acf_explanation"] = str(expl).strip()[:200]

    lang = result.get("answer_lang")
    if lang is None or not str(lang).strip():
        query = result.get("query")
        if isinstance(query, str) and query.strip():
            try:
                from ml.rag.chatbot.answer_language import detect_answer_language

                lang = detect_answer_language(query)
            except Exception:
                lang = None
    if lang is not None and str(lang).strip():
        out["answer_lang"] = str(lang).strip()
    return out


def summarize_rag_result_for_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Retrieval/rerank counts and soft-fail flags for root trace output metadata."""
    route = infer_rag_route(result)
    news_n = len(result.get("vector_news_results") or [])
    academic_papers_n = len(result.get("vector_academic_papers_results") or [])
    policies_n = len(result.get("vector_policies_results") or [])
    public_reports_n = len(result.get("vector_public_reports_results") or [])
    formation_n = len(result.get("vector_formation_results") or [])
    academic_n = len(result.get("vector_academic_results") or []) or (
        academic_papers_n + policies_n + public_reports_n + formation_n
    )
    ota_n = len(result.get("vector_ota_results") or [])
    bq_n = len(result.get("bq_results") or [])
    web_n = len(result.get("web_results") or [])
    answer = str(result.get("answer") or "").strip()
    is_shortcut = bool(
        result.get("is_meta_query")
        or result.get("is_product_query")
        or result.get("is_help_query")
        or result.get("is_greeting_query")
        or result.get("is_out_of_scope_query")
        or result.get("is_language_unknown")
    )
    empty_retrieval = (not is_shortcut) and (
        news_n + academic_papers_n + policies_n + public_reports_n + formation_n + ota_n + bq_n + web_n == 0
    )
    web_status = result.get("web_fallback_status")
    summary: dict[str, Any] = {
        "route": route,
        "vector_news_count": news_n,
        "vector_academic_papers_count": academic_papers_n,
        "vector_policies_count": policies_n,
        "vector_public_reports_count": public_reports_n,
        "vector_formation_count": formation_n,
        "vector_academic_count": academic_n,
        "vector_ota_count": ota_n,
        "bq_table_candidates_count": len(result.get("bq_table_candidates") or []),
        "bq_results_count": bq_n,
        "merged_context_count": len(result.get("merged_context") or []),
        "reranked_context_count": len(result.get("reranked_context") or []),
        "web_results_count": web_n,
        "citation_count": len(result.get("citations") or []),
        "empty_retrieval": empty_retrieval,
        "llm_empty_answer": (not is_shortcut) and not answer,
        **_bq_soft_fail_flags(result),
        **_acf_and_language_for_trace(result),
    }
    if web_status is not None:
        summary["web_fallback_status"] = str(web_status)
        summary["web_fallback_used"] = str(web_status).lower() not in ("", "skipped", "none", "off")
    rerank_mode = result.get("rerank_mode") or result.get("_rerank_mode")
    if rerank_mode:
        summary["rerank_mode"] = str(rerank_mode)
    if result.get("error"):
        summary["error"] = str(result.get("error"))[:200]
    if result.get("latency_ms") is not None:
        summary["latency_ms"] = float(result["latency_ms"])
    task_mode = result.get("task_mode")
    if task_mode:
        summary["task_mode"] = str(task_mode)
    for key in (
        "early_short_circuit",
        "skipped_decompose_llm",
        "skipped_retrieval",
        "decompose_llm_ms",
        "vector_ms",
        "corpus_count",
        "cascade_level",
        "bq_nl2sql_ms",
        "bq_execute_ms",
        "sql_source",
        "rerank_ms",
        "rerank_pool_size",
        "generate_ms",
        "generate_max_tokens",
        "generate_input_chars",
        "generate_input_tokens",
        "bq_timeout",
        "route_candidate",
    ):
        if result.get(key) is not None:
            summary[key] = result.get(key)
    corpus_sel = result.get("corpus_selection")
    if isinstance(corpus_sel, dict) and corpus_sel.get("active"):
        summary["corpus_count"] = summary.get("corpus_count") or len(corpus_sel.get("active") or [])
    chars = summary.get("generate_input_chars")
    if summary.get("generate_input_tokens") is None and chars is not None:
        try:
            summary["generate_input_tokens"] = int((int(chars) + 3) // 4)
        except (TypeError, ValueError):
            pass
    return summary


def _record_soft_fail_scores(result: dict[str, Any], summary: dict[str, Any]) -> None:
    """Emit boolean Langfuse scores only when soft-fail flags are true."""
    tid = get_current_trace_id()
    if not tid:
        return
    flags = (
        ("empty_retrieval", bool(summary.get("empty_retrieval"))),
        ("llm_empty_answer", bool(summary.get("llm_empty_answer"))),
        (
            "bq_failure",
            bool(summary.get("bq_validation_failed") or summary.get("bq_execution_failed")),
        ),
        ("web_fallback_used", bool(summary.get("web_fallback_used"))),
    )
    for name, on in flags:
        if on:
            record_trace_score(name=name, value=True, trace_id=tid)


def _record_acf_score(summary: dict[str, Any]) -> None:
    """Emit Path B ACF score (0–100) on the root trace when present."""
    raw = summary.get("acf_score")
    if raw is None:
        return
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return
    record_trace_score(name="acf_score", value=score)


def infer_rag_route(result: dict[str, Any]) -> str:
    """Derive pipeline route label from a ``run_rag()`` result dict."""
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
        tag_list = _build_tags(
            route=route_label,
            answer_lang=str(retrieval_summary.get("answer_lang") or "") or None,
            acf_band=str(retrieval_summary.get("acf_band") or "") or None,
        )
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
        try:
            _record_soft_fail_scores(result, retrieval_summary)
        except Exception:
            pass
        try:
            _record_acf_score(retrieval_summary)
        except Exception:
            pass


@contextmanager
def rag_trace_context(
    *,
    trace_name: str = "rag.query",
    session_id: str | None = None,
    user_id: str | None = None,
    plan_type: str | None = None,
    category: str | None = None,
    trace_input: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[RagTraceHandle]:
    """
    Root span for one RAG request with session/user/plan/category propagated to children.

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
    uid = (user_id or "").strip()
    if uid:
        propagate_kwargs["user_id"] = uid
    if tag_list:
        propagate_kwargs["tags"] = tag_list

    # Only swallow failures *starting* the root observation. Exceptions from the
    # caller's body must propagate — a second yield in ``except`` triggers
    # ``RuntimeError: generator didn't stop after throw()``.
    try:
        observation = client.start_as_current_observation(
            as_type="span",
            name=trace_name,
            input=trace_input,
        )
    except Exception:
        with openrouter_run_context(fallback_run_id):
            yield handle
        return

    with observation as root_span:
        handle.span = root_span
        run_id = get_current_trace_id() or fallback_run_id
        try:
            meta = {"openrouter_session_id": run_id}
            if uid:
                meta["user_id"] = uid
            release = tracing_release()
            if release:
                meta["release"] = release
            root_span.update_trace(metadata=meta)
        except Exception:
            pass
        with openrouter_run_context(run_id):
            if propagate_kwargs:
                with propagate_attributes(**propagate_kwargs):
                    yield handle
            else:
                yield handle
        handle._closed = True


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
    """
    Return a lazy Langfuse ``@observe`` proxy.

    Resolves at *call* time so modules that bind
    ``_observe = get_observe_decorator()`` at import still work when
    ``LANGFUSE_*`` keys are loaded later (dotenv / Railway).
    """

    def _lazy_observe(*obs_args: Any, **obs_kwargs: Any) -> Callable[[Any], Any]:
        def decorator(fn: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if observe is not None and is_tracing_enabled():
                    return observe(*obs_args, **obs_kwargs)(fn)(*args, **kwargs)
                return fn(*args, **kwargs)

            wrapper.__name__ = getattr(fn, "__name__", "wrapped")
            wrapper.__qualname__ = getattr(fn, "__qualname__", wrapper.__name__)
            wrapper.__doc__ = getattr(fn, "__doc__", None)
            return wrapper

        return decorator

    return _lazy_observe


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
    # Only swallow failures *starting* the observation. Body exceptions must
    # propagate — yielding again in ``except`` causes
    # ``RuntimeError: generator didn't stop after throw()``.
    try:
        observation = client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
        )
    except Exception:
        yield None
        return
    with observation as span:
        yield span


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
