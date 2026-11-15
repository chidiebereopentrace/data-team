"""Graph routing tests for early non-RAG short-circuit."""

from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.graph import node_decompose
from ml.rag.chatbot.active_request import resolve_turn


def test_node_decompose_greeting_skips_memory_and_decompose() -> None:
    with mock.patch("ml.rag.chatbot.graph.enrich_query_to_english") as enrich:
        with mock.patch("ml.rag.chatbot.graph.build_query_views") as views:
            with mock.patch("ml.rag.chatbot.graph.decompose_query") as decompose:
                with mock.patch("ml.rag.chatbot.graph.observed_span") as span:
                    with mock.patch("ml.rag.chatbot.graph.update_current_span_metadata") as meta:
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
    assert out.get("early_short_circuit") is True
    assert out.get("skipped_retrieval") is True
    assert out.get("route_candidate") == "greeting"
    assert out.get("language_enrich_applied") is False
    enrich.assert_not_called()
    views.assert_not_called()
    decompose.assert_not_called()
    meta.assert_called_once()
    flags = meta.call_args[0][0]
    assert flags.get("early_short_circuit") is True
    assert flags.get("skipped_decompose_llm") is True
    assert flags.get("skipped_retrieval") is True


def test_node_decompose_igbo_question_not_greeting_and_enriches(monkeypatch) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    q = "kedu ihe mere ka mmeputa akpu n'obodo niajiria gbawe kemgbe afo ise ugbua?"
    from ml.rag.chatbot.language_enrich import LanguageEnrichment

    with mock.patch(
        "ml.rag.chatbot.graph.enrich_query_to_english",
        return_value=LanguageEnrichment(
            answer_lang="ig",
            needs_enrich=True,
            english_query="What caused cassava production in Nigeria to change over the last five years?",
            enrich_source="llm",
        ),
    ) as enrich:
        with mock.patch("ml.rag.chatbot.graph.decompose_query") as decompose:
            decompose.return_value = {
                "intent": "analytical",
                "entities": ["cassava"],
                "geography": ["Nigeria"],
                "domains": ["production"],
                "time_start": "2020-01-01",
                "time_end": "2025-12-31",
                "_decompose_llm_ms": 0.0,
                "_skipped_decompose_llm": True,
            }
            with mock.patch("ml.rag.chatbot.graph.observed_span") as span:
                span.return_value.__enter__ = mock.Mock(return_value=None)
                span.return_value.__exit__ = mock.Mock(return_value=False)
                out = node_decompose({"query": q})
    enrich.assert_called_once()
    decompose.assert_called_once()
    # Decompose planning text should be the English enrich.
    assert "cassava" in decompose.call_args[0][0].lower()
    assert out.get("is_greeting_query") is not True
    assert out.get("answer_lang") == "ig"
    assert out.get("language_enrich_applied") is True
    assert "cassava" in str(out.get("language_enrich") or "").lower()
    assert out.get("query") == out.get("user_query")
    assert "kedu" in str(out.get("user_query") or "").lower()


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


def test_node_decompose_incident_query_short_circuits() -> None:
    incident = "what is your use, and what can i use AskADZA for"
    with mock.patch("ml.rag.chatbot.graph.build_query_views") as views:
        with mock.patch("ml.rag.chatbot.graph.decompose_query") as decompose:
            with mock.patch("ml.rag.chatbot.graph.observed_span") as span:
                with mock.patch("ml.rag.chatbot.graph.update_current_span_metadata") as meta:
                    span.return_value.__enter__ = mock.Mock(return_value=None)
                    span.return_value.__exit__ = mock.Mock(return_value=False)
                    out = node_decompose(
                        {
                            "query": incident,
                            "recent_turns": [
                                {"role": "user", "content": "maize yields in Kenya 2020"},
                            ],
                        }
                    )
    assert out["is_product_query"] is True
    assert out["is_help_query"] is True
    assert out.get("early_short_circuit") is True
    assert out.get("skipped_decompose_llm") is True
    assert out.get("skipped_retrieval") is True
    assert out.get("route_candidate") == "help"
    assert out.get("decompose_llm_ms") == 0.0
    views.assert_not_called()
    decompose.assert_not_called()
    meta.assert_called_once()
    flags = meta.call_args[0][0]
    assert flags.get("route_candidate") == "help"
    assert flags.get("early_short_circuit") is True
    assert flags.get("skipped_decompose_llm") is True
    assert flags.get("skipped_retrieval") is True


def test_social_followups_never_carry_slots() -> None:
    prior = [{"role": "user", "content": "maize yields in Kenya 2020"}]
    for msg in ["hi", "merci", "asante", "sannu", "kedu", "sawubona"]:
        turn, _ = resolve_turn(msg, persisted=None, recent_turns=prior)
        assert turn.carried() == {}, msg


def test_node_decompose_agri_query_runs_full_decompose() -> None:
    with mock.patch("ml.rag.chatbot.graph.decompose_query") as decompose:
        decompose.return_value = {
            "intent": "descriptive",
            "entities": ["maize"],
            "geography": ["Kenya"],
            "domains": ["production"],
            "time_start": "2020-01-01",
            "time_end": "2020-12-31",
            "_decompose_llm_ms": 0.0,
            "_skipped_decompose_llm": True,
        }
        with mock.patch("ml.rag.chatbot.graph.observed_span") as span:
            span.return_value.__enter__ = mock.Mock(return_value=None)
            span.return_value.__exit__ = mock.Mock(return_value=False)
            out = node_decompose({"query": "maize production in Kenya 2020"})
    decompose.assert_called_once()
    assert out.get("is_greeting_query") is not True
    assert out.get("task_mode") != "chat" or out.get("decomposition")
