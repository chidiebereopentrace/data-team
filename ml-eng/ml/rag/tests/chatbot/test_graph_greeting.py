"""Graph routing tests for early non-RAG short-circuit."""

from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.graph import node_decompose
from ml.rag.chatbot.query_enricher import enrich_query_with_memory


def test_node_decompose_greeting_skips_memory_and_decompose() -> None:
    with mock.patch("ml.rag.chatbot.graph.enrich_query_with_memory") as enrich:
        with mock.patch("ml.rag.chatbot.graph.decompose_query") as decompose:
            with mock.patch("ml.rag.chatbot.graph.observed_span") as span:
                span.return_value.__enter__ = mock.Mock(return_value=None)
                span.return_value.__exit__ = mock.Mock(return_value=False)
                out = node_decompose(
                    {
                        "query": "hi",
                        "recent_turns": [
                            {"role": "user", "content": "maize yields in Kenya 2020"},
                        ],
                    }
                )
    assert out["is_greeting_query"] is True
    assert out.get("task_mode") == "chat"
    enrich.assert_not_called()
    decompose.assert_not_called()


def test_node_decompose_meta_skips_decompose() -> None:
    with mock.patch("ml.rag.chatbot.graph.decompose_query") as decompose:
        with mock.patch("ml.rag.chatbot.graph.observed_span") as span:
            span.return_value.__enter__ = mock.Mock(return_value=None)
            span.return_value.__exit__ = mock.Mock(return_value=False)
            out = node_decompose({"query": "Who are you?"})
    assert out["is_meta_query"] is True
    decompose.assert_not_called()


def test_node_decompose_product_skips_decompose() -> None:
    with mock.patch("ml.rag.chatbot.graph.decompose_query") as decompose:
        with mock.patch("ml.rag.chatbot.graph.observed_span") as span:
            span.return_value.__enter__ = mock.Mock(return_value=None)
            span.return_value.__exit__ = mock.Mock(return_value=False)
            out = node_decompose({"query": "What is OpenTrace?"})
    assert out["is_product_query"] is True
    decompose.assert_not_called()


def test_enricher_never_merges_social_followups() -> None:
    prior = [{"role": "user", "content": "maize yields in Kenya 2020"}]
    for msg in ["hi", "merci", "asante", "sannu", "kedu", "sawubona"]:
        out = enrich_query_with_memory(msg, recent_turns=prior)
        assert out["enriched"] is False, msg
        assert out["enriched_query"] == msg, msg
