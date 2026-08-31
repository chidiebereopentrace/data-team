"""
Verify Hugging Face credentials used by the RAG stack.

From repo root:

  PYTHONPATH=. python -m ml.rag.check_hf

Loads ``data/local/.env`` and ``config/.env`` (same rules as ``local_env.load_rag_dotenv``).

1. ``whoami-v2`` — token is valid for the Hub.
2. Chat completion — ``huggingface_hub.InferenceClient.chat.completions`` (non-stream),
   same as NL-to-SQL / decomposer / reranker / memory.

Env:

- ``HF_API_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` / ``HF_TOKEN``
- ``RAG_LLM_MODEL_ID`` (default from ``llm_chat.DEFAULT_LLM_MODEL_ID``)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

from ml.rag.hf_chat import hf_chat_sync
from ml.rag.hf_token import get_hf_api_token
from ml.rag.llm_chat import llm_model_id
from ml.rag.local_env import load_rag_dotenv


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    load_rag_dotenv(repo_root)

    token = get_hf_api_token()
    if not token:
        print(
            "No Hugging Face token found. Set one of: HF_API_TOKEN, HUGGINGFACE_HUB_TOKEN, HF_TOKEN "
            "(e.g. in data/local/.env).",
            file=sys.stderr,
        )
        return 2

    print("1) Hub token (whoami-v2)…")
    r = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    print(f"   HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"   {r.text[:800]}", file=sys.stderr)
        print(
            "\nFix: create a token at https://huggingface.co/settings/tokens "
            "with read access; for Inference / router usage you may need additional scopes.",
            file=sys.stderr,
        )
        return 1
    try:
        data = r.json()
        name = data.get("name") or data.get("fullname") or "?"
        print(f"   OK — logged in as {name!r}")
    except Exception:
        print("   OK — response was not JSON (unexpected but HTTP 200).")

    model = llm_model_id()
    print(
        f"\n2) InferenceClient chat (non-stream; generator uses streaming) — model={model!r}…"
    )
    out = hf_chat_sync(
        [{"role": "user", "content": "Reply with exactly: OK"}],
        model=model,
        max_tokens=16,
        temperature=0.0,
    )
    if out:
        print(f"   OK — sample reply: {out!r}")
        print("\nHF Inference API path used by this repo should work for chat completions.")
        return 0

    print(
        "   Empty response or InferenceClient raised (token, model access, or provider error).",
        file=sys.stderr,
    )
    print(
        "\nCommon causes:\n"
        "  • Missing/invalid token — set HF_API_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_TOKEN.\n"
        "  • Gated model — accept the license on the model card.\n"
        "  • HF inference billing (402) — add credits at huggingface.co/settings/billing, "
        "or set RAG_LLM_BASE_URL to a local OpenAI-compatible server (LM Studio).\n"
        "  • Model not available on your inference provider — try another RAG_LLM_MODEL_ID.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
