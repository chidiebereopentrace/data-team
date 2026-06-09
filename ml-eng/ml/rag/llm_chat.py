"""
Unified chat-completions client for the RAG stack.

Supports:
- OpenAI-compatible servers via ``RAG_LLM_BASE_URL`` (OpenRouter, DeepInfra, LM Studio, vLLM)
- Hugging Face router (fallback when ``HF_API_TOKEN`` is set and ``RAG_LLM_BASE_URL`` is unset)

Never raises on HTTP/API errors — returns empty string so callers can fall back.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import requests

from ml.rag.hf_token import get_hf_api_token

logger = logging.getLogger(__name__)

HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# Billing, auth, capacity — treat as soft failures (no exception to LangGraph).
_SOFT_FAIL_HTTP = frozenset({401, 402, 403, 410, 429, 502, 503})

_usage_lock = threading.Lock()
_request_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


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
    with _usage_lock:
        _request_usage["prompt_tokens"] = 0
        _request_usage["completion_tokens"] = 0
        _request_usage["total_tokens"] = 0


def get_llm_usage() -> TokenUsage:
    with _usage_lock:
        return TokenUsage(
            prompt_tokens=_request_usage["prompt_tokens"],
            completion_tokens=_request_usage["completion_tokens"],
            total_tokens=_request_usage["total_tokens"],
        )


def add_llm_usage(raw: dict[str, Any] | None) -> None:
    """Accumulate OpenAI-style usage from a chat completion response."""
    if not raw or not isinstance(raw, dict):
        return
    prompt = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    completion = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    total = int(raw.get("total_tokens") or (prompt + completion))
    with _usage_lock:
        _request_usage["prompt_tokens"] += prompt
        _request_usage["completion_tokens"] += completion
        _request_usage["total_tokens"] += total


def llm_model_id() -> str:
    return os.environ.get("RAG_LLM_MODEL_ID", "meta-llama-3.1-8b-instruct").strip()


def llm_default_timeout_s() -> float:
    """Default HTTP timeout for local LLM servers (LM Studio can be slow on large prompts)."""
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


def llm_uses_hf_router() -> bool:
    url = llm_chat_completions_url() or ""
    return "router.huggingface.co" in url


def llm_uses_openrouter() -> bool:
    url = llm_chat_completions_url() or ""
    return "openrouter.ai" in url


def _llm_api_key() -> str:
    """Bearer token for OpenAI-compatible backends (OpenRouter, DeepInfra, LM Studio)."""
    for key in ("RAG_LLM_API_KEY", "OPENROUTER_API_KEY"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return ""


def _openrouter_extra_headers() -> dict[str, str]:
    """Optional OpenRouter attribution headers (not required for API to work)."""
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_APP_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def llm_chat_complete(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_s: float | None = None,
) -> str:
    """
    Non-streaming chat completion. Returns assistant text or ``""`` on failure.

    Use OpenAI-style ``system`` / ``user`` messages — do not embed Llama chat templates
    inside ``user`` content when calling LM Studio (it applies the template itself).
    """
    url = llm_chat_completions_url()
    if not url:
        logger.warning("llm_chat_complete: no backend (set RAG_LLM_BASE_URL or HF_API_TOKEN)")
        return ""

    effective_timeout = llm_default_timeout_s() if timeout_s is None else timeout_s
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
        local_key = _llm_api_key()
        if local_key:
            headers["Authorization"] = f"Bearer {local_key}"
        headers.update(_openrouter_extra_headers())

    payload = {
        "model": model or llm_model_id(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=effective_timeout)
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
        add_llm_usage(data.get("usage") if isinstance(data, dict) else None)
        choices = data.get("choices") or []
        if not choices:
            logger.warning("llm_chat_complete: empty choices from %s", url)
            return ""
        content = choices[0].get("message", {}).get("content")
        if content is None:
            logger.warning("llm_chat_complete: null message content from %s", url)
            return ""
        return str(content).strip()
    except requests.Timeout:
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
