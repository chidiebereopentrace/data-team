"""
Unified chat-completions client for the RAG stack.

All LLM calls go through HTTP to an OpenAI-compatible API:

- **Production (recommended):** OpenRouter via ``RAG_LLM_BASE_URL=https://openrouter.ai/api/v1``
  and ``RAG_LLM_API_KEY``.
- **Local dev:** LM Studio / vLLM via ``RAG_LLM_BASE_URL`` (e.g. ``http://127.0.0.1:1234/v1``).
- **Fallback:** Hugging Face router when ``HF_API_TOKEN`` is set and no base URL is configured.

Never raises on HTTP/API errors — returns empty string so callers can fall back.
"""
from __future__ import annotations

import logging
import os
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import requests

from ml.rag.chatbot.stage_budget import RetrievalWorkers, remaining_seconds
from ml.rag.hf_token import get_hf_api_token
from ml.rag.observability import (
    get_observe_decorator,
    get_openrouter_run_id,
    openrouter_sessions_enabled,
    safe_llm_trace_input,
    update_current_llm_generation,
)

_observe = get_observe_decorator()

logger = logging.getLogger(__name__)

HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# Billing, auth, capacity — treat as soft failures (no exception to LangGraph).
_SOFT_FAIL_HTTP = frozenset({401, 402, 403, 410, 429, 502, 503})

_usage_lock = threading.Lock()
_request_usage: ContextVar[dict[str, int] | None] = ContextVar("rag_llm_usage", default=None)
_HTTP_WORKERS = RetrievalWorkers("rag-llm-http", workers=12)

# Per-request truncation tracking (ML-054). OpenAI-compatible APIs report
# finish_reason=="length" when generation stopped because max_tokens was hit.
# That signal was previously read and discarded, so multi-part answers could be
# cut off mid-sentence with no error and no indication to the user.
_truncation_lock = threading.Lock()


@dataclass
class _Truncation:
    length_capped_calls: int = 0
    last_finish_reason: str = ""


_request_truncation: ContextVar[_Truncation | None] = ContextVar("rag_llm_truncation", default=None)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
        }


def reset_llm_usage() -> None:
    """Clear per-request token accumulator (call at start of run_rag)."""
    _request_usage.set({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def get_llm_usage() -> TokenUsage:
    with _usage_lock:
        return TokenUsage(**(_request_usage.get() or {}))


def reset_llm_truncation() -> None:
    """Clear per-request truncation state (call at start of run_rag)."""
    _request_truncation.set(_Truncation())


def record_finish_reason(reason: str | None) -> None:
    """Record an OpenAI-style finish_reason from a chat completion response."""
    r = (reason or "").strip().lower()
    if not r:
        return
    truncation = _request_truncation.get()
    if truncation is None:
        truncation = _Truncation()
        _request_truncation.set(truncation)
    with _truncation_lock:
        truncation.last_finish_reason = r
        if r == "length":
            truncation.length_capped_calls += 1


def last_finish_reason() -> str:
    with _truncation_lock:
        truncation = _request_truncation.get()
        return truncation.last_finish_reason if truncation else ""


def answer_was_length_capped() -> bool:
    """True when the most recent completion stopped on the max_tokens ceiling."""
    return last_finish_reason() == "length"


def length_capped_call_count() -> int:
    with _truncation_lock:
        truncation = _request_truncation.get()
        return truncation.length_capped_calls if truncation else 0


def add_llm_usage(raw: dict[str, Any] | None) -> None:
    """Accumulate OpenAI-style usage from a chat completion response."""
    if not raw or not isinstance(raw, dict):
        return
    prompt = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    completion = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    total = int(raw.get("total_tokens") or (prompt + completion))
    usage = _request_usage.get()
    if usage is None:
        reset_llm_usage()
        usage = _request_usage.get()
    assert usage is not None
    with _usage_lock:
        usage["prompt_tokens"] += prompt
        usage["completion_tokens"] += completion
        usage["total_tokens"] += total


DEFAULT_LLM_MODEL_ID = "qwen/qwen3-30b-a3b-instruct-2507"


def llm_model_id() -> str:
    return os.environ.get("RAG_LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID).strip() or DEFAULT_LLM_MODEL_ID


def llm_default_timeout_s() -> float:
    """Default HTTP timeout for the LLM backend (OpenRouter, LM Studio, etc.)."""
    return float(os.environ.get("RAG_LLM_TIMEOUT_S", "180") or 180)


def llm_configured() -> bool:
    return llm_chat_completions_url() is not None


def llm_chat_completions_url() -> str | None:
    """Return chat completions URL, or None when no backend is configured."""
    base = os.environ.get("RAG_LLM_BASE_URL", "").strip().rstrip("/")
    if base:
        return f"{base}/chat/completions"
    if get_hf_api_token():
        return HF_ROUTER_CHAT_URL
    return None


def llm_uses_openrouter() -> bool:
    base = os.environ.get("RAG_LLM_BASE_URL", "").strip().lower()
    return "openrouter.ai" in base


def llm_uses_hf_router() -> bool:
    url = llm_chat_completions_url() or ""
    return "router.huggingface.co" in url


@_observe(as_type="generation", name="llm_chat_complete")
def llm_chat_complete(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_s: float | None = None,
    purpose: str | None = None,
) -> str:
    """
    Non-streaming chat completion. Returns assistant text or ``""`` on failure.

    Use OpenAI-style ``system`` / ``user`` messages. Managed APIs (OpenRouter) and
    local OpenAI-compatible servers (LM Studio) apply chat templates server-side.
    """
    effective_model = model or llm_model_id()
    gen_metadata = {"purpose": purpose} if purpose else None
    update_current_llm_generation(
        input_data=safe_llm_trace_input(messages, effective_model),
        model=effective_model,
        metadata=gen_metadata,
    )
    url = llm_chat_completions_url()
    if not url:
        logger.warning("llm_chat_complete: no backend (set RAG_LLM_BASE_URL or HF_API_TOKEN)")
        return ""

    effective_timeout = llm_default_timeout_s() if timeout_s is None else timeout_s
    effective_timeout = remaining_seconds(effective_timeout)
    if effective_timeout == 0:
        logger.info("LLM request skipped after the RAG deadline")
        return ""
    headers: dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if llm_uses_hf_router():
        token = get_hf_api_token()
        if not token:
            return ""
        headers["Authorization"] = f"Bearer {token}"
    else:
        api_key = os.environ.get("RAG_LLM_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    if llm_uses_openrouter():
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
        title = os.environ.get("OPENROUTER_APP_TITLE", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

    payload = {
        "model": effective_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if openrouter_sessions_enabled() and llm_uses_openrouter():
        run_id = get_openrouter_run_id()
        if run_id:
            payload["session_id"] = run_id[:256]
            headers["x-session-id"] = run_id[:256]
    future = None
    try:
        future = _HTTP_WORKERS.try_submit(
            lambda: requests.post(url, headers=headers, json=payload, timeout=effective_timeout)
        )
        if future is None:
            logger.info("LLM request skipped while HTTP workers are busy")
            return ""
        resp = future.result(timeout=remaining_seconds(effective_timeout))
        if resp.status_code in _SOFT_FAIL_HTTP:
            logger.warning(
                "llm_chat_complete: HTTP %s from %s (model=%s)",
                resp.status_code,
                url,
                payload["model"],
            )
            return ""
        resp.raise_for_status()
        data = resp.json()
        usage_raw = data.get("usage") if isinstance(data, dict) else None
        add_llm_usage(usage_raw if isinstance(usage_raw, dict) else None)
        choices = data.get("choices") or []
        if not choices:
            logger.warning("llm_chat_complete: empty choices from %s", url)
            return ""
        # ML-054: capture finish_reason so a max_tokens cut-off is no longer
        # silently discarded (multi-part answers were ending mid-sentence).
        record_finish_reason(choices[0].get("finish_reason"))
        content = choices[0].get("message", {}).get("content")
        if content is None:
            logger.warning("llm_chat_complete: null message content from %s", url)
            return ""
        text = str(content).strip()
        update_current_llm_generation(
            output=text,
            model=effective_model,
            usage=usage_raw if isinstance(usage_raw, dict) else None,
            metadata=gen_metadata,
        )
        return text
    except (requests.Timeout, TimeoutError):
        if future is not None:
            future.cancel()
        logger.warning(
            "llm_chat_complete: timed out after %.0fs (model=%s, url=%s)",
            effective_timeout,
            payload["model"],
            url,
        )
        return ""
    except Exception:
        logger.exception("llm_chat_complete: request failed (model=%s, url=%s)", payload["model"], url)
        return ""
