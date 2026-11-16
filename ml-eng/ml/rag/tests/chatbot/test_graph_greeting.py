"""Graph routing tests for early non-RAG short-circuit."""

from __future__ import annotations

from unittest import mock

import pytest

from ml.rag.chatbot.graph import _document_search_texts, node_decompose
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
    assert out.get("is_greeting_query") is True
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


@pytest.mark.parametrize(
    ("query", "lang", "commodity"),
    [
        ("ego ole ka ha na-ele osikapa na rwanda n'afo 2025", "ig", "rice"),
        ("bei ya mahindi nchini Rwanda mwaka 2025", "sw", "maize"),
        ("menene farashin shinkafa a Rwanda a shekarar 2025", "ha", "rice"),
    ],
)
def test_local_price_query_recovers_from_english_language_label(
    monkeypatch: pytest.MonkeyPatch, query: str, lang: str, commodity: str
) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    english = f"What was the price of {commodity} in Rwanda in 2025?"
    with (
        mock.patch(
            "ml.rag.chatbot.answer_language._commonlingua_detect_with_confidence",
            return_value=("en", 0.85),
        ),
        mock.patch(
            "ml.rag.chatbot.language_enrich.llm_chat_complete", return_value=english
        ) as translate,
        mock.patch(
            "ml.rag.chatbot.query_decomposer._decompose_backend_configured", return_value=False
        ),
    ):
        out = node_decompose({"query": query})

    assert out.get("answer_lang") == lang
    translate.assert_called_once()
    assert out.get("route_candidate") == "full_rag"
    decomposition = out["decomposition"]
    assert commodity in decomposition["entities"]
    assert decomposition["geography"] == ["Rwanda"]
    assert decomposition["time_start"] == "2025-01-01"
    assert out["user_query"] == query


def test_clear_english_price_query_skips_translation_when_lid_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    with (
        mock.patch(
            "ml.rag.chatbot.answer_language._commonlingua_detect_with_confidence", return_value=None
        ),
        mock.patch("ml.rag.chatbot.language_enrich.llm_chat_complete", return_value="") as translate,
        mock.patch(
            "ml.rag.chatbot.query_decomposer._decompose_backend_configured", return_value=False
        ),
    ):
        out = node_decompose({"query": "whats the price of maize in kenya this year"})

    assert out.get("answer_lang") == "en"
    translate.assert_not_called()


def test_policy_impact_plans_diagnosis_without_second_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    query = "How did agricultural policies affect farmer livelihoods and the price of food in West Africa since 2015?"
    with (
        mock.patch(
            "ml.rag.chatbot.answer_language._commonlingua_detect_with_confidence",
            return_value=("en", 0.85),
        ),
        mock.patch(
            "ml.rag.chatbot.query_decomposer._decompose_backend_configured", return_value=False
        ),
        mock.patch("ml.rag.chatbot.metacog.llm_chat_complete", return_value="") as advisory_llm,
    ):
        out = node_decompose({"query": query})

    assert out.get("task_mode") == "analytical"
    assert out["turn_contract"]["job"] == "diagnose"
    plan = out.get("metacog_plan")
    assert plan and "research" in plan["evidence_needs"]
    advisory_llm.assert_not_called()


def test_multipart_question_retains_distinct_search_topics_and_measures() -> None:
    query = "How did fertilizer subsidies affect farm incomes, maize prices and rice production in Ghana?"
    with (
        mock.patch(
            "ml.rag.chatbot.answer_language._commonlingua_detect_with_confidence",
            return_value=("en", 0.85),
        ),
        mock.patch(
            "ml.rag.chatbot.query_decomposer._decompose_backend_configured", return_value=True
        ),
        mock.patch(
            "ml.rag.chatbot.query_decomposer._call_llama_decompose",
            return_value={
                "job": "fact",
                "geography": ["Ghana"],
                "entities": ["maize", "rice"],
                "primary_measure_hints": ["market_price", "production"],
                "evidence_topics": ["fertilizer subsidies", "farm incomes", "invented policy"],
            },
        ),
    ):
        state = node_decompose({"query": query})

    decomposition = state["decomposition"]
    assert {"market_price", "production"}.issubset(decomposition["primary_measures"])
    assert state["turn_contract"]["job"] == "diagnose"
    assert _document_search_texts(state) == [query, "fertilizer subsidies", "farm incomes"]


def test_node_decompose_igbo_question_not_greeting_and_enriches(monkeypatch) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    q = "kedu ihe mere ka mmeputa akpu n'obodo niajiria gbawe kemgbe afo ise ugbua?"
    from ml.rag.chatbot.language_enrich import LanguageEnrichment

    with mock.patch(
        "ml.rag.chatbot.graph.detect_language_with_confidence",
        return_value=("ig", 0.92),
    ):
        with mock.patch(
            "ml.rag.chatbot.graph.enrich_query_to_english",
            return_value=LanguageEnrichment(
                answer_lang="ig",
                needs_enrich=True,
                english_query=(
                    "What caused cassava production in Nigeria to change "
                    "over the last five years?"
                ),
                enrich_source="llm",
                confidence=0.92,
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
    assert "cassava" in decompose.call_args[0][0].lower()
    assert out.get("is_greeting_query") is not True
    assert out.get("answer_lang") == "ig"
    assert out.get("language_enrich_applied") is True
    assert "cassava" in str(out.get("language_enrich") or "").lower()
    assert out.get("query") == out.get("user_query")
    assert "kedu" in str(out.get("user_query") or "").lower()


def test_node_decompose_osikapa_enriches_without_flag(monkeypatch) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    q = "ego ole ka ana-ele osikapa na kenya"
    from ml.rag.chatbot.language_enrich import LanguageEnrichment

    with mock.patch(
        "ml.rag.chatbot.graph.detect_language_with_confidence",
        return_value=("ig", 0.88),
    ):
        with mock.patch(
            "ml.rag.chatbot.graph.enrich_query_to_english",
            return_value=LanguageEnrichment(
                answer_lang="ig",
                needs_enrich=True,
                english_query="What is the price of rice in Kenya?",
                enrich_source="llm",
                confidence=0.88,
            ),
        ) as enrich:
            with mock.patch("ml.rag.chatbot.graph.decompose_query") as decompose:
                decompose.return_value = {
                    "intent": "descriptive",
                    "entities": ["rice"],
                    "geography": ["Kenya"],
                    "domains": ["market"],
                    "primary_measures": ["market_price"],
                    "_decompose_llm_ms": 0.0,
                    "_skipped_decompose_llm": True,
                }
                with mock.patch("ml.rag.chatbot.graph.observed_span") as span:
                    span.return_value.__enter__ = mock.Mock(return_value=None)
                    span.return_value.__exit__ = mock.Mock(return_value=False)
                    out = node_decompose({"query": q})
    enrich.assert_called_once()
    assert "rice" in decompose.call_args[0][0].lower()
    assert out.get("answer_lang") == "ig"
    assert out.get("language_enrich_applied") is True
    assert out.get("user_query") == q or "osikapa" in str(out.get("user_query") or "")
    assert out.get("is_greeting_query") is not True


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
