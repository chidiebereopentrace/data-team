#!/usr/bin/env python3
"""
Verify Langfuse tracing configuration for the OpenTrace RAG stack.

Usage (from ml-eng/ with PYTHONPATH=.):
  python scripts/verify_langfuse_tracing.py
  python scripts/verify_langfuse_tracing.py --smoke   # optional live query (needs full RAG env)

Phase 0 checklist (EU Cloud UI):
  1. Who are you?           -> route meta
  2. What is OpenTrace?     -> route product
  3. Maize yields in Kenya 2020 -> route full_rag

Confirm each trace shows: root span, LangGraph nodes, nested llm_chat_complete, retrieval.* spans.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.rag.local_env import load_rag_dotenv
from ml.rag.observability import flush_langfuse, get_langfuse_client, is_tracing_enabled

PRESET_QUERIES = [
    ("meta", "Who are you?"),
    ("product", "What is OpenTrace?"),
    ("full_rag", "Maize yields in Kenya 2020"),
]


def _check_env() -> list[str]:
    missing: list[str] = []
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        if not os.environ.get(key, "").strip():
            missing.append(key)
    base = os.environ.get("LANGFUSE_BASE_URL", "").strip() or os.environ.get("LANGFUSE_HOST", "").strip()
    if not base:
        missing.append("LANGFUSE_BASE_URL (or LANGFUSE_HOST)")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Langfuse RAG tracing setup")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one preset full_rag query (requires Qdrant + LLM env)",
    )
    args = parser.parse_args()

    load_rag_dotenv(_REPO)
    missing = _check_env()
    if missing:
        print("Langfuse tracing: NOT CONFIGURED")
        print("Add to ml-eng/config/.env (local) and Railway/GCE service variables:")
        for key in missing:
            print(f"  - {key}")
        print("\nExample:")
        print("  LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("  LANGFUSE_SECRET_KEY=sk-lf-...")
        print("  LANGFUSE_BASE_URL=https://cloud.langfuse.com")
        print("  LANGFUSE_TRACING_ENVIRONMENT=development")
        return 1

    print("Langfuse tracing: keys present")
    print(f"  base_url: {os.environ.get('LANGFUSE_BASE_URL') or os.environ.get('LANGFUSE_HOST')}")
    env = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", "").strip()
    if env:
        print(f"  environment tag: {env}")
    from ml.rag.observability import tracing_release

    release = tracing_release()
    if release:
        print(f"  release tag: {release}")
    else:
        print("  release tag: (unset — set LANGFUSE_TRACING_RELEASE or RAILWAY_GIT_COMMIT_SHA)")

    if not is_tracing_enabled():
        print("ERROR: is_tracing_enabled() is False despite keys — check langfuse package.")
        return 1

    client = get_langfuse_client()
    if client is None:
        print("ERROR: get_langfuse_client() returned None — SDK init failed.")
        return 1

    print("Langfuse client: OK")
    print("\nManual UI verification (https://cloud.langfuse.com):")
    for route, query in PRESET_QUERIES:
        print(f"  [{route}] {query!r}")
    print("  Expect nested: decompose/merge/web_fallback, retrieval.*, rerank, llm purpose tags")
    print("  Optional: pass user_id on /query for Langfuse Users")

    if args.smoke:
        from ml.rag.graph import run_rag
        from ml.rag.observability import rag_trace_context

        route, query = PRESET_QUERIES[2]
        print(f"\nSmoke: running {route!r} query...")
        with rag_trace_context(
            trace_name="rag.verify",
            session_id="langfuse-verify",
            user_id="verify-user",
            trace_input={"query": query},
            tags=["verify"],
        ) as handle:
            result = run_rag(query, trace_tags=["verify"])
            handle.update_output(result)
        flush_langfuse()
        print(f"  answer length: {len(str(result.get('answer') or ''))}")
        print("  Check Langfuse UI for trace name rag.verify (user verify-user)")

    flush_langfuse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
