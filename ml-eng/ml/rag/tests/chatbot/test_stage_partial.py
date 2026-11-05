"""Partial answers when a stage overruns, and the multilingual rerank default."""
from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.generator import generate
from ml.rag.chatbot.stage_budget import stage_over_budget
from ml.rag.rerank_client import DEFAULT_RERANK_MODEL_ID, rerank_model_id


def test_stage_budget_flags_overrun(monkeypatch) -> None:
    monkeypatch.setenv("RAG_STAGE_RETRIEVE_BUDGET_S", "2")
    assert stage_over_budget(3.0, env_var="RAG_STAGE_RETRIEVE_BUDGET_S", default_s=12.0)
    assert not stage_over_budget(1.0, env_var="RAG_STAGE_RETRIEVE_BUDGET_S", default_s=12.0)


def test_openrouter_rerank_default_is_multilingual(monkeypatch) -> None:
    monkeypatch.delenv("RAG_RERANK_MODEL_ID", raising=False)
    assert DEFAULT_RERANK_MODEL_ID == "cohere/rerank-multilingual-v3.0"
    assert rerank_model_id() == "cohere/rerank-multilingual-v3.0"


def test_generate_returns_source_text_when_model_times_out() -> None:
    item = {
        "content": "Maize prices in Kenya rose through the 2024 season.",
        "source": "news",
        "_context_kind": "news",
        "metadata": {
            "title": "Kenya maize brief",
            "publisher": "FEWS NET",
            "published_at": "2024-06-01",
        },
    }
    with mock.patch("ml.rag.chatbot.generator._call_llama", return_value=""):
        result = generate(
            "what happened to maize prices in Kenya",
            [item],
            task_mode="chat",
            decomposition={"geography": ["Kenya"]},
        )
    assert "Kenya maize brief" in result.answer or "Maize prices in Kenya" in result.answer
    assert result.citations
