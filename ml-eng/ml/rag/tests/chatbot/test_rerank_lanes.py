"""Warehouse rows are packed as structured evidence and are not passage-reranked."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.graph import node_rerank


def test_warehouse_rows_are_not_sent_to_the_passage_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def _fake_rerank(query: str, items: list[dict], **kwargs: object) -> list[dict]:
        seen["sources"] = [str(item.get("source")) for item in items]
        return list(items)

    monkeypatch.setattr("ml.rag.chatbot.graph.rerank", _fake_rerank)
    documents = [
        {"content": f"passage {i} about markets", "source": "academic", "_context_kind": "academic", "score": 0.2}
        for i in range(12)
    ]
    state = {
        "query": "yam production trend in Ghana and Nigeria",
        "task_mode": "chat",
        "corpus_selection": {"active": ["academic", "public_reports"]},
        "merged_context": [
            {"content": "Ghana 10 Nigeria 20", "source": "bigquery", "_context_kind": "bigquery"},
            *documents,
        ],
        "turn_contract": {"job": "trend", "serve_status": "served", "measure_id": "production"},
    }
    out = node_rerank(state)
    assert "bigquery" not in seen["sources"]
    assert any(item.get("source") == "bigquery" for item in out["reranked_context"])
    assert out["retrieval_stage_counts"]["structured"] == 1
