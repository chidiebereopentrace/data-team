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
from typing import Any

import requests

from ml.rag.hf_token import get_hf_api_token

# Local model support
_LOCAL_MODEL_CACHE: dict[str, Any] = {}
_LOCAL_TOKENIZER_CACHE: dict[str, Any] = {}

try:
    from langfuse.decorators import observe  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    def observe(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn
        return decorator

logger = logging.getLogger(__name__)

HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# Billing, auth, capacity — treat as soft failures (no exception to LangGraph).
_SOFT_FAIL_HTTP = frozenset({401, 402, 403, 410, 429, 502, 503})


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


@observe(as_type="generation", capture_input=True, capture_output=True)
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
    # Check if we should use local transformers model
    if _use_local_model():
        model_id = model or llm_model_id()
        logger.info("Using local transformers model: %s", model_id)
        return _local_model_generate(messages, model_id, max_tokens, temperature)
    
    # Otherwise use API (HuggingFace router or local server)
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
