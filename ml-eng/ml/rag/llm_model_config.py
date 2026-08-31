"""Resolved LLM model slugs for observability (/ready, startup logs)."""
from __future__ import annotations

import logging
import os
from typing import Any

from ml.rag.chatbot.plan_policy import model_for_plan, valid_plan_type_ids
from ml.rag.llm_chat import DEFAULT_LLM_MODEL_ID, llm_model_id

logger = logging.getLogger(__name__)

_ROUTER_MODEL_PREFIX = "openrouter/"


def _summary_model_id() -> str:
    return (os.environ.get("RAG_SUMMARY_MODEL_ID") or "").strip() or llm_model_id()


def resolved_llm_models() -> dict[str, Any]:
    """Return effective model slugs per pipeline purpose (non-secret)."""
    from ml.rag.chatbot.bq_sql_reasoner import _reasoner_model
    from ml.rag.retrievers.bq_retriever import _nl2sql_model_id

    plan_models = {
        plan_id: model_for_plan(plan_id) or llm_model_id()
        for plan_id in sorted(valid_plan_type_ids())
    }
    return {
        "chat_default": llm_model_id(),
        "summary": _summary_model_id(),
        "bq_nl2sql": _nl2sql_model_id(),
        "bq_reasoner": _reasoner_model("Government"),
        "plan_models": plan_models,
        "code_default": DEFAULT_LLM_MODEL_ID,
    }


def warn_router_models(models: dict[str, Any] | None = None) -> list[str]:
    """Log WARNING for OpenRouter router slugs (unpredictable backend routing)."""
    snapshot = models if models is not None else resolved_llm_models()
    router_slugs: list[str] = []

    def _check(slug: str, label: str) -> None:
        s = (slug or "").strip()
        if s.lower().startswith(_ROUTER_MODEL_PREFIX):
            router_slugs.append(f"{label}={s!r}")

    _check(str(snapshot.get("chat_default") or ""), "chat_default")
    _check(str(snapshot.get("summary") or ""), "summary")
    _check(str(snapshot.get("bq_nl2sql") or ""), "bq_nl2sql")
    _check(str(snapshot.get("bq_reasoner") or ""), "bq_reasoner")
    for plan_id, slug in (snapshot.get("plan_models") or {}).items():
        _check(str(slug), f"plan_models.{plan_id}")

    if router_slugs:
        logger.warning(
            "OpenRouter router models detected (backend model may differ from slug in logs): %s. "
            "Set explicit RAG_LLM_MODEL_ID / RAG_BQ_* model IDs for predictable routing.",
            "; ".join(router_slugs),
        )
    return router_slugs


def log_resolved_llm_models() -> dict[str, Any]:
    """Log resolved model slugs at INFO; warn on router models."""
    models = resolved_llm_models()
    warn_router_models(models)
    logger.info(
        "LLM models: chat_default=%s summary=%s bq_nl2sql=%s bq_reasoner=%s",
        models["chat_default"],
        models["summary"],
        models["bq_nl2sql"],
        models["bq_reasoner"],
    )
    for plan_id in sorted((models.get("plan_models") or {}).keys()):
        logger.info("LLM plan_models.%s=%s", plan_id, models["plan_models"][plan_id])
    return models


__all__ = [
    "log_resolved_llm_models",
    "resolved_llm_models",
    "warn_router_models",
]
