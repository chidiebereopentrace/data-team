"""A supported-language question keeps its subject through enrichment and routing."""
from unittest.mock import patch
import pytest

from ml.rag.chatbot.answer_language import generation_failure_answer
from ml.rag.chatbot.language_enrich import clear_language_enrich_cache, enrich_query_to_english
from ml.rag.chatbot.task_mode import resolve_task_mode
from ml.rag.chatbot.streamlit_inspector import normalize_query_response
from ml.rag.chatbot.graph import node_decompose

QUESTION = "kedu ihe iwere ikwu banyere onudo aja alaugbo di na kenya"


def test_subject_replacement_is_not_accepted_as_useful_expansion():
    clear_language_enrich_cache()
    with patch("ml.rag.chatbot.language_enrich.llm_chat_complete", return_value=(
        "What were the factors influencing the adoption of improved maize varieties in Kenya?"
    )):
        result = enrich_query_to_english(QUESTION, answer_lang="ig", confidence=0.9)
    assert "soil" in result.english_query.lower()
    assert "maize" not in result.english_query.lower()
    assert "kenya" in result.english_query.lower()


@pytest.mark.parametrize(("question", "language"), [
    (QUESTION, "ig"), ("Expliquez l'état des sols au Kenya", "fr"), ("Hali ya udongo Kenya", "sw"),
])
def test_missing_translation_retains_subject_and_cached_guard(question, language):
    clear_language_enrich_cache()
    with patch("ml.rag.chatbot.language_enrich.llm_chat_complete", return_value="") as llm:
        first = enrich_query_to_english(question, answer_lang=language, confidence=0.9)
        second = enrich_query_to_english(question, answer_lang=language, confidence=0.9)
    assert first.subject_guard_applied and second.subject_guard_applied
    assert first.subjects == second.subjects == ("soil",)
    assert question in first.english_query and "soil" in first.english_query
    assert second.enrich_source == "cache" and llm.call_count == 1


def test_expansion_can_add_related_context_while_retaining_the_subject():
    clear_language_enrich_cache()
    expansion = "Assess soil conditions in Kenya, including implications for maize yields."
    with patch("ml.rag.chatbot.language_enrich.llm_chat_complete", return_value=expansion):
        result = enrich_query_to_english(QUESTION, answer_lang="ig", confidence=0.9)
    assert result.english_query == expansion


def test_farmland_context_alone_does_not_replace_an_employment_subject():
    clear_language_enrich_cache()
    question = "kedu ụgwọ ọrụ ndi ọrụ n'ala ugbo na Kenya"
    translation = "What are agricultural workers' wages in Kenya?"
    with patch("ml.rag.chatbot.language_enrich.llm_chat_complete", return_value=translation):
        result = enrich_query_to_english(question, answer_lang="ig", confidence=0.9)
    assert result.english_query == translation
    assert not result.subject_guard_applied


def test_broad_diagnosis_is_not_given_a_single_fact_generation_budget():
    assert resolve_task_mode("Assess soil conditions in Kenya", {
        "job": "diagnose", "intent": "diagnostic", "geography": ["Kenya"],
        "primary_measures": ["soil"],
    }) == "analytical"


def test_igbo_generation_failure_stays_in_igbo():
    assert generation_failure_answer("ig") != generation_failure_answer("en")
    assert "Biko" in generation_failure_answer("ig")


def test_http_inspector_keeps_the_actual_enrichment_and_query_views():
    trace = {"language_enrich": "Assess soil conditions in Kenya", "language_enrich_applied": True,
             "generate_finish_reason": "timeout",
             "task_mode": "analytical", "generation_plan": {"answer_shape": "comparison"},
             "retrieval_stage_counts": {"candidate": 134, "packed": 18}, "evidence_drops": {"over_top_k": 116},
             "query_views": {"user_query": QUESTION, "language_enrich": "Assess soil conditions in Kenya"}}
    result = normalize_query_response({"answer": "ok", "trace": trace}, query=QUESTION, latency_ms=1, kwargs={})
    assert result["language_enrich"] == trace["language_enrich"]
    assert result["language_enrich_applied"]
    assert result["query_views"] == trace["query_views"]
    assert result["generate_finish_reason"] == "timeout"
    for key in ("task_mode", "generation_plan", "retrieval_stage_counts", "evidence_drops"):
        assert result[key] == trace[key]


def test_rejected_subject_facets_do_not_become_saved_commodity_slots(monkeypatch):
    clear_language_enrich_cache()
    with patch("ml.rag.chatbot.language_enrich.llm_chat_complete", return_value=(
        "What were the factors influencing the adoption of improved maize varieties in Kenya?"
    )):
        enrichment = enrich_query_to_english(QUESTION, answer_lang="ig", confidence=0.9)
    monkeypatch.setattr("ml.rag.chatbot.graph.detect_language_with_confidence", lambda q: ("ig", 0.9))
    monkeypatch.setattr("ml.rag.chatbot.graph.enrich_query_to_english", lambda *a, **kw: enrichment)
    monkeypatch.setattr("ml.rag.chatbot.graph.decompose_query", lambda *a, **kw: {
        "intent": "diagnostic", "job": "diagnose", "entities": ["maize"], "geography": ["Kenya"],
        "primary_measure_hints": ["production", "yield", "soil"],
        "evidence_topics": ["adoption of improved maize varieties"],
    })
    monkeypatch.setattr("ml.rag.chatbot.graph.metacog_enabled", lambda: False)
    result = node_decompose({"query": QUESTION, "reference_date": "2026-11-20"})
    assert result["answer_lang"] == "ig"
    assert result["language_enrich_subject_guard"]
    assert result["task_mode"] == "analytical"
    assert {fact["measure_id"] for fact in result["turn_contract"]["facts"]} == {"soil"}
    assert "maize" not in result["decomposition"]["entities"]
    assert "commodity" not in result["active_request"]["slots"]
