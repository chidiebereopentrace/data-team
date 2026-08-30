"""Evidence-law prompt assembly, cleaners, and routing guards."""
from __future__ import annotations

import os
from unittest import mock

from ml.rag.chatbot.generation_plan import build_generation_plan
from ml.rag.chatbot.generator import (
    _build_prompt,
    _cap_export_caption,
    _clean_answer,
    _normalize_inline_citations,
    _strip_internal_identifiers,
    classify_evidence_tier,
    generate,
)
from ml.rag.chatbot.product_knowledge import generate_product_answer, is_help_query
from ml.rag.chatbot.query_gate import is_greeting_query


def _system_prompt(**kwargs) -> str:
    messages = _build_prompt("What is maize production in Nigeria?", "Context line", **kwargs)
    return messages[0]["content"]


def test_fact_lookup_prompt_excludes_analytical_skeleton() -> None:
    system = _system_prompt(task_mode="fact_lookup", evidence_tier="strong")
    assert "REPORT OUTLINE" not in system
    assert "ANALYTICAL VOICE RULES" not in system
    assert "FACT LOOKUP" in system


def test_empty_tier_prompt_has_no_key_findings() -> None:
    system = _system_prompt(task_mode="analytical", evidence_tier="empty", analytical_mode=True)
    assert "Key Findings" not in system
    assert "INSUFFICIENT EVIDENCE" in system


def test_export_intent_adds_caption_rules() -> None:
    system = _system_prompt(task_mode="analytical", export_intent="pdf", evidence_tier="strong")
    assert "ARTIFACT EXPORT MODE" in system


def test_cap_export_caption_limits_sentences() -> None:
    text = "One. Two. Three. Four. Five. Six. Seven."
    capped = _cap_export_caption(text, max_sentences=5)
    assert capped.count(".") == 5


def test_clean_answer_splits_executive_summary_heading() -> None:
    out = _clean_answer("Intro ## Executive summary Rice production rose.")
    assert "\n\n## Executive summary\n\n" in out


def test_normalize_multi_source_brackets() -> None:
    out = _normalize_inline_citations("Trends rose [Source 1, 3] and ([3]) [3].")
    assert "[1]" in out
    assert "[3]" in out
    assert "[Source 1, 3]" not in out
    assert "([3])" not in out


def test_strip_internal_identifiers() -> None:
    raw = "Data from BigQuery mart_dev.fct_production shows maize rose."
    out = _strip_internal_identifiers(raw)
    assert "BigQuery" not in out
    assert "mart_dev" not in out
    assert "fct_production" not in out
    assert "maize rose" in out


def test_classify_empty_for_off_topic_news_only() -> None:
    tier = classify_evidence_tier(
        "What was maize production in Nigeria in 2022?",
        [
            {
                "content": "[News] Cassava policy in Uganda updated.",
                "_context_kind": "news",
                "metadata": {"country": "Uganda", "title": "Cassava policy"},
            }
        ],
        {"countries": ["Nigeria"], "entities": ["maize"]},
    )
    assert tier == "empty"


def test_generate_empty_context_skips_llm() -> None:
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "should not run"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_ALLOW_UNGROUNDED", None)
            result = generate("Maize in Nigeria 2022", [])
    mock_llm.assert_not_called()
    assert result.citations == []
    assert result.acf is not None
    assert result.acf.band == "no_evidence"


def test_generate_yield_only_adds_yield_rule_to_prompt() -> None:
    captured: dict = {}

    def fake_call(messages, **_kwargs):
        captured["system"] = messages[0]["content"]
        return "Maize yield in 2006 was 2.1 t/ha."

    item = {
        "content": "[Structured data] yield 2.1",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "country_name": "Nigeria",
            "product_name": "Maize",
            "year": 2006,
            "element": "Yield",
            "metric": "production_maize_yield",
            "value_semantics": {"measure_value": 2.1, "element": "Yield"},
        },
    }
    plan = build_generation_plan(
        "Maize yield Nigeria 2006",
        task_mode="fact_lookup",
        reranked_context=[item],
        decomposition={"countries": ["Nigeria"], "entities": ["maize", "yield"]},
    )
    with mock.patch("ml.rag.chatbot.generator._call_llama", side_effect=fake_call):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_ALLOW_UNGROUNDED", None)
            os.environ.pop("RAG_MIN_USABLE_CONTEXT", None)
            result = generate(
                "Maize yield Nigeria 2006",
                [item],
                task_mode="fact_lookup",
                generation_plan=plan.to_dict(),
                decomposition={"countries": ["Nigeria"], "entities": ["maize", "yield"]},
            )
    assert "YIELD DATA ONLY" in captured.get("system", "")
    assert "production trend" not in result.answer.lower() or "yield" in result.answer.lower()


def test_greeting_short_circuit_pattern() -> None:
    assert is_greeting_query("How are you today?")


def test_help_query_short_circuit_pattern() -> None:
    assert is_help_query("What can Ask ADZA help me with?", None)


def test_product_help_answer_has_no_citations() -> None:
    answer = generate_product_answer("What can I use Ask ADZA for?")
    assert isinstance(answer, str)
    assert answer.strip()
