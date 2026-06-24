"""
Unified chat-completions client for the RAG stack.

Supports:
- Local transformers models (when transformers is installed and no API URL set)
- Hugging Face router (default when ``HF_API_TOKEN`` is set)
- OpenAI-compatible local servers (LM Studio, vLLM) via ``RAG_LLM_BASE_URL``

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

# Local model support
_LOCAL_MODEL_CACHE: dict[str, Any] = {}
_LOCAL_TOKENIZER_CACHE: dict[str, Any] = {}

try:
    from langfuse.decorators import observe
except Exception:  # pragma: no cover
    def observe(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn
        return decorator

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


def _use_local_model() -> bool:
    """Check if we should use local transformers instead of API."""
    # Explicit provider setting takes precedence
    provider = os.environ.get("RAG_LLM_PROVIDER", "").strip().lower()
    if provider == "local":
        return True
    if provider in ("openai", "hf_api"):
        return False

    # If there's an explicit API URL, use API
    if os.environ.get("RAG_LLM_BASE_URL", "").strip():
        return False

    # If transformers is available, use local
    # (HF token can be used to download models)
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _load_local_model(model_id: str) -> Any:
    """Load local transformers model with caching."""
    if model_id in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_id]

    try:
        from transformers import AutoModelForCausalLM
        import torch

        # Get HF token for authentication (needed for gated models)
        hf_token = get_hf_api_token()

        logger.info("Loading local model: %s (this may take 2-3 minutes on first call)", model_id)
        logger.info("Downloading model weights if not cached locally...")

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        # Build kwargs, only include token if it exists
        model_kwargs = {
            "low_cpu_mem_usage": True,
        }

        # Only use float16 on GPU, use float32 on CPU
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        if hf_token:  # Only pass token if it exists and is not empty
            model_kwargs["token"] = hf_token

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        # Move to device if not using device_map
        if device == "cpu":
            model = model.to(device)

        _LOCAL_MODEL_CACHE[model_id] = model
        logger.info("Local model loaded successfully: %s on %s", model_id, device)
        return model
    except Exception as e:
        logger.exception("Failed to load local model %s: %s", model_id, e)
        return None


def _load_local_tokenizer(model_id: str) -> Any:
    """Load local tokenizer with caching."""
    if model_id in _LOCAL_TOKENIZER_CACHE:
        return _LOCAL_TOKENIZER_CACHE[model_id]

    try:
        from transformers import AutoTokenizer

        # Get HF token for authentication (needed for gated models)
        hf_token = get_hf_api_token()

        # Build kwargs, only include token if it exists
        tokenizer_kwargs = {}
        if hf_token:  # Only pass token if it exists and is not empty
            tokenizer_kwargs["token"] = hf_token

        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        _LOCAL_TOKENIZER_CACHE[model_id] = tokenizer
        return tokenizer
    except Exception as e:
        logger.exception("Failed to load tokenizer for %s: %s", model_id, e)
        return None


def _local_model_generate(
    messages: list[dict[str, Any]],
    model_id: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate text using local transformers model."""
    model = _load_local_model(model_id)
    tokenizer = _load_local_tokenizer(model_id)

    if model is None or tokenizer is None:
        logger.warning("Local model or tokenizer not available for %s", model_id)
        return ""

    try:
        import torch

        logger.info("Generating response with local model: %s", model_id)

        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.info("Prompt length: %d characters", len(prompt))

        # Tokenize and move to correct device
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        logger.info("Input tokens: %d, generating up to %d new tokens...", input_ids.shape[1], max_tokens)

        # Generate (no context manager needed - model.device is just a device object)
        with torch.no_grad():  # Save memory
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.1,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (remove prompt)
        if prompt in generated:
            response = generated[len(prompt):].strip()
        else:
            response = generated.strip()

        logger.info("Generated response length: %d characters", len(response))
        return response
    except Exception as e:
        logger.exception("Local model generation failed for %s: %s", model_id, e)
        return ""


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
        local_key = os.environ.get("RAG_LLM_API_KEY", "").strip()
        if local_key:
            headers["Authorization"] = f"Bearer {local_key}"

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
