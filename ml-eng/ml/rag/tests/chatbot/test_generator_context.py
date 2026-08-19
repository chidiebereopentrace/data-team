"""Unit tests for generator context packing and citation filtering."""

from __future__ import annotations

import os
from unittest import mock

from ml.rag.chatbot.context_diversity import dedupe_context_items
from ml.rag.chatbot.generator import (
    GenerationResult,
    SourceRef,
    _append_structured_citations,
    _build_context_block,
    _build_prompt,
    _clean_answer,
    _citation_url,
    _context_max_chars,
    _finalize_generation_result,
    _format_source_citation,
    _drop_geo_conflicting,
    _generate_max_tokens,
    _no_data_fallback_message,
    _normalize_inline_citations,
    _registry_from_context_items,
    _strip_invalid_citation_markers,
    _strip_model_sources_appendix,
    _strip_preamble_openers,
    dedupe_bq_context_items,
    extract_referenced_source_ids,
    filter_context_items,
    generate,
    is_usable_context_item,
    referenced_citations,
)
from ml.rag.text_processors.preprocess.bibliographic_metadata import format_academic_citation


def _bq_item() -> dict:
    return {
        "content": "[Structured data] {'country': 'Senegal', 'product': 'rice', 'yield': 2.1}",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "sql": "SELECT * FROM `opentrace.raw_dev.yield_raw_data` LIMIT 10",
            "country": "Senegal",
            "product": "rice",
            "planting_year": 2020,
        },
    }


def _news_item() -> dict:
    return {
        "content": "[News] Rice policy update in Senegal",
        "source": "news",
        "_context_kind": "news",
        "metadata": {
            "doc_kind": "news_article",
            "title": "Senegal rice policy shift",
            "publisher": "AgriNews",
            "published_at": "2023-06-01",
        },
    }


def test_generate_max_tokens_default_and_env() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_GENERATE_MAX_TOKENS", None)
        assert _generate_max_tokens("fact_lookup") == 384
        assert _generate_max_tokens("analytical") == 1280
        assert _generate_max_tokens("fact_lookup") < _generate_max_tokens("analytical")
    with mock.patch.dict(os.environ, {"RAG_GENERATE_MAX_TOKENS": "512"}):
        assert _generate_max_tokens("analytical") == 512


def test_context_max_chars_memory_reduces_budget() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_GENERATE_CONTEXT_MAX_CHARS", None)
        assert _context_max_chars("", task_mode="analytical") == 12000
        assert _context_max_chars("", task_mode="fact_lookup") == 6000
        assert _context_max_chars("x" * 2000, task_mode="analytical") == 10000


def test_build_prompt_export_intent_uses_caption_not_analytical_brief() -> None:
    msgs = _build_prompt(
        "export maize production Kenya as csv",
        context_block="[Source 1] data",
        task_mode="analytical",
        analytical_mode=True,
        export_intent="csv",
    )
    system = msgs[0]["content"]
    assert "ARTIFACT EXPORT MODE" in system or "caption" in system.lower()
    assert "ANALYTICAL BRIEF MODE" not in system


def test_build_context_block_respects_budget() -> None:
    items = [_bq_item(), _news_item()] + [
        {
            "content": f"[Academic] chunk {i} " + "x" * 500,
            "_context_kind": "academic",
            "metadata": {"doc_kind": "academic_article", "authors": f"Author {i}", "publication_year": "2020"},
        }
        for i in range(8)
    ]
    block, registry = _build_context_block(items, budget=5000, chunk_cap=2000)
    assert len(block) <= 5000 + 50
    assert "[Source 1]" in block
    assert "Type:" in block
    assert "Citation:" in block
    assert "Unknown authors" not in block
    assert len(registry) >= 2


def test_build_context_block_rank_weighting() -> None:
    items = [
        {"content": "first " + "a" * 1000, "_context_kind": "news", "metadata": {"title": "First"}},
        {"content": "second " + "b" * 1000, "_context_kind": "news", "metadata": {"title": "Second"}},
    ]
    block, _ = _build_context_block(items, budget=3000, chunk_cap=3000)
    assert block.index("[Source 1") < block.index("[Source 2")
    assert len(block.split("first")[1].split("[Source 2]")[0]) >= len(block.split("second")[1])


def test_bq_citation_format() -> None:
    line = _format_source_citation(_bq_item())
    assert line is not None
    assert "[Structured data]" in line
    assert "OpenTrace agricultural data" in line
    assert "yield_raw_data" not in line
    assert "Senegal" in line or "rice" in line


def test_bq_citation_uses_faostat_when_table_is_definitive() -> None:
    item = _bq_item()
    item["metadata"]["table_id"] = "stg_faostat_production"
    item["metadata"]["sql"] = (
        "SELECT * FROM `opentrace-prod-5ga4.staging_dev.stg_faostat_production`"
    )
    line = _format_source_citation(item)
    assert line is not None
    assert line.startswith("[FAOSTAT]")
    assert "[Structured data]" not in line
    assert "stg_faostat" not in line


def test_bq_citation_uses_domain_when_table_id_is_generic() -> None:
    item = _bq_item()
    item["metadata"]["sql"] = "SELECT * FROM `opentrace.raw_dev.yield_raw_data`"
    item["metadata"]["source_domain"] = "fews_net"
    line = _format_source_citation(item)
    assert line is not None
    assert line.startswith("[FEWS NET]")
    assert "[Structured data]" not in line


def _bq_error_item() -> dict:
    return {
        "content": "[BQ execution error: 404 Not found: Table raw_dev.fews_net]",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "sql": "SELECT * FROM `opentrace.raw_dev.fews_net_food_security_master`",
            "execution_error": "404 Not found",
        },
    }


def test_filter_bq_failure_chunks() -> None:
    assert not is_usable_context_item(_bq_error_item())
    filtered = filter_context_items([_bq_item(), _bq_error_item(), _news_item()])
    assert len(filtered) == 2
    assert filtered[0]["source"] == "bigquery"
    assert filtered[1]["source"] == "news"


def test_strip_model_sources_appendix() -> None:
    raw = (
        "Rice intensification works.[18] More policy support.[19]\n\n"
        "Limitations: partial coverage.\n\n"
        "Sources:\n[1]\nType: Structured data\n[BQ execution error: 404]\n\n"
        "[18]\nType: News\n[News] headline"
    )
    prose = _strip_model_sources_appendix(raw)
    assert "Sources:" not in prose
    assert "BQ execution error" not in prose
    assert "[18]" in prose
    assert extract_referenced_source_ids(raw) == {18, 19}


def test_append_citations_ignores_model_sources_dump() -> None:
    registry = [
        SourceRef(i, _news_item(), f"[News] Item {i}") for i in range(1, 20)
    ]
    answer = (
        "Policy helped rice.[14] News confirms.[18] Sector outlook.[19]\n\n"
        "Sources:\n" + "\n".join(f"[{i}]\nType: News" for i in range(1, 20))
    )
    with mock.patch.dict(os.environ, {"RAG_CITATIONS_MODE": "referenced"}):
        out = _append_structured_citations(answer, registry)
    assert "14. [News]" in out
    assert "18. [News]" in out
    assert "19. [News]" in out
    assert "1. [News]" not in out
    assert "BQ execution" not in out
    assert "Type: Structured data" not in out
    assert out.count("Sources") == 1


def test_extract_referenced_source_ids() -> None:
    text = "Rice yields rose [1] while policy shifted per Source 3."
    assert extract_referenced_source_ids(text) == {1, 3}


def test_normalize_verbose_inline_citations() -> None:
    raw = (
        "From [Source 5 | Academic | Unknown authors (2019)], yields rose [Source 3] "
        "and policy shifted per Source 9."
    )
    out = _normalize_inline_citations(raw)
    assert "Unknown authors" not in out
    assert "[Source 5" not in out
    assert "[5]" in out
    assert "[3]" in out
    assert "[9]" in out
    assert extract_referenced_source_ids(out) == {3, 5, 9}


def test_normalize_preserves_named_prose_citations() -> None:
    raw = (
        "According to Branca et al. (2012), agriculture's GDP share rose.[6] "
        "Business News Nigeria (2025) reports regional price gaps.[9]"
    )
    out = _normalize_inline_citations(raw)
    assert "Branca et al. (2012)" in out
    assert "Business News Nigeria (2025)" in out
    assert extract_referenced_source_ids(out) == {6, 9}


def test_format_academic_citation_prefers_title_over_unknown_authors() -> None:
    cite = format_academic_citation(
        {"publication_year": "2019", "article_title": "Rice market dynamics in Nigeria"}
    )
    assert "Unknown authors" not in cite
    assert "Rice market dynamics" in cite
    assert "2019" in cite


def test_append_citations_referenced_only() -> None:
    registry = [
        SourceRef(1, _news_item(), "[News] Senegal rice policy shift — AgriNews (2023)"),
        SourceRef(2, _bq_item(), "[Structured data] OpenTrace agricultural data (country=Senegal)"),
    ]
    answer = "Production trends improved [1] in recent seasons."
    with mock.patch.dict(os.environ, {"RAG_CITATIONS_MODE": "referenced"}):
        out = _append_structured_citations(answer, registry)
    assert "Sources" in out
    assert "1. [News]" in out
    assert "2. [Structured data]" not in out


def test_append_citations_all_mode() -> None:
    registry = [
        SourceRef(1, _news_item(), "[News] Senegal rice policy shift — AgriNews (2023)"),
        SourceRef(2, _bq_item(), "[Structured data] OpenTrace agricultural data"),
    ]
    answer = "Production trends improved."
    with mock.patch.dict(os.environ, {"RAG_CITATIONS_MODE": "all"}):
        out = _append_structured_citations(answer, registry)
    assert "1. [News]" in out
    assert "2. [Structured data]" in out


def test_web_wikipedia_citation_format() -> None:
    item = {
        "content": "[Wikipedia | Agriculture in Senegal] summary text",
        "_context_kind": "web_wikipedia",
        "metadata": {
            "title": "Agriculture in Senegal",
            "url": "https://en.wikipedia.org/wiki/Agriculture_in_Senegal",
        },
    }
    line = _format_source_citation(item)
    assert line is not None
    assert "[Wikipedia]" in line
    assert "Agriculture in Senegal" in line
    assert "wikipedia.org" in line


def test_web_search_citation_format() -> None:
    item = {
        "content": "[Web | Senegal rice policy] snippet",
        "_context_kind": "web_search",
        "metadata": {"title": "Senegal rice policy", "url": "https://example.com/news"},
    }
    line = _format_source_citation(item)
    assert line is not None
    assert "[Web]" in line
    assert "example.com" in line


def test_append_citations_no_refs_when_unreferenced() -> None:
    registry = [SourceRef(1, _news_item(), "[News] Title")]
    answer = "Production trends improved with no inline cites."
    with mock.patch.dict(os.environ, {"RAG_CITATIONS_MODE": "referenced"}):
        out = _append_structured_citations(answer, registry)
    assert "Sources" in out
    assert "1. [News]" in out


def test_referenced_citations_structured_shape() -> None:
    registry = [
        SourceRef(1, _news_item(), "[News] Senegal rice policy shift — AgriNews (2023)"),
        SourceRef(2, _bq_item(), "[Structured data] OpenTrace agricultural data"),
    ]
    answer = "Production trends improved [1] in recent seasons."
    with mock.patch.dict(os.environ, {"RAG_CITATIONS_MODE": "referenced"}):
        cites = referenced_citations(answer, registry)
    assert len(cites) == 1
    assert cites[0]["id"] == 1
    assert cites[0]["kind"] == "news"
    assert cites[0]["text"].startswith("[News]")
    assert cites[0]["url"] is None


def test_generate_returns_generation_result_without_sources_block() -> None:
    registry_items = [_news_item(), _bq_item()]
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "Rice policy shifted.[1]"
        with mock.patch.dict(
            os.environ,
            {"RAG_CITATIONS_MODE": "referenced", "RAG_APPEND_SOURCES_TO_ANSWER": ""},
            clear=False,
        ):
            os.environ.pop("RAG_APPEND_SOURCES_TO_ANSWER", None)
            result = generate("What about rice?", registry_items)
    assert isinstance(result, GenerationResult)
    assert "Sources" not in result.answer
    assert "[1]" not in result.answer
    assert "Rice policy shifted." in result.answer
    # Default chat: packed sources populate citations[] without inline footnotes.
    assert len(result.citations) == 2


# ---------------------------------------------------------------------------
# Sprint 1 (Jul 2026) — no-data fallback + temperature + ungrounded guard
# ---------------------------------------------------------------------------


def test_no_data_fallback_contains_acf_marker_and_gap_block() -> None:
    """Test-1 finding: gap responses were invisible. Every fallback must carry ACF signal."""
    msg = _no_data_fallback_message(
        "What are cocoa yields in Ghana in 2024?",
        decomposition={"countries": ["Ghana"], "time_start": "2024-01-01", "time_end": "2024-12-31"},
    )
    assert "I don't have OpenTrace data" in msg
    assert "**Gap**" in msg
    assert "Query: What are cocoa yields in Ghana in 2024?" in msg
    assert "Ghana" in msg
    assert "2024-01-01" in msg and "2024-12-31" in msg
    assert "**What would help**" in msg
    assert "ACF: no evidence" in msg


def test_no_data_fallback_handles_missing_decomposition() -> None:
    """Fallback must still be well-formed without geo/time hints."""
    msg = _no_data_fallback_message("random unrelated question", decomposition=None)
    assert "I don't have OpenTrace data" in msg
    assert "Query: random unrelated question" in msg
    assert "Geography:" not in msg  # no geo hint → no line
    assert "Time period:" not in msg  # no time hint → no line
    assert "ACF: no evidence" in msg
    assert "What would help" in msg


def test_no_data_fallback_handles_empty_query() -> None:
    msg = _no_data_fallback_message("", decomposition=None)
    assert "Query: your question" in msg
    assert "ACF: no evidence" in msg


def test_generate_empty_context_returns_structured_fallback() -> None:
    """Sprint 1: replaces old single-sentence 'couldn't find' string."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_ALLOW_UNGROUNDED", None)
        result = generate("What is the price of rice in Mars?", [])
    assert isinstance(result, GenerationResult)
    assert result.citations == []
    assert "I don't have OpenTrace data" in result.answer
    assert "ACF: no evidence" in result.answer
    assert "confirm the knowledge bases are loaded" not in result.answer  # old copy is gone


def test_generate_empty_context_fallback_includes_decomposition_hints() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_ALLOW_UNGROUNDED", None)
        result = generate(
            "Rice yields in Kenya 2023",
            [],
            decomposition={"countries": ["Kenya"], "time_period": "2023"},
        )
    assert "Kenya" in result.answer
    assert "2023" in result.answer
    assert "ACF: no evidence" in result.answer


def test_call_llama_default_temperature() -> None:
    """
    Sprint 1: default set to 0.7 for response variety across similar queries.
    Synthesis-on-gaps protection is provided by code guardrails (fallback path,
    hardened ungrounded prompt, chunk filtering), not by low temperature.
    """
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_GENERATE_TEMPERATURE", None)
        with mock.patch("ml.rag.chatbot.generator.llm_chat_complete") as mock_complete:
            mock_complete.return_value = "ok"
            from ml.rag.chatbot.generator import _call_llama

            _call_llama([{"role": "user", "content": "hi"}])
        _, kwargs = mock_complete.call_args
        assert kwargs["temperature"] == 0.7


def test_call_llama_temperature_env_override_respected() -> None:
    with mock.patch.dict(os.environ, {"RAG_GENERATE_TEMPERATURE": "0.1"}):
        with mock.patch("ml.rag.chatbot.generator.llm_chat_complete") as mock_complete:
            mock_complete.return_value = "ok"
            from ml.rag.chatbot.generator import _call_llama

            _call_llama([{"role": "user", "content": "hi"}])
        _, kwargs = mock_complete.call_args
        assert kwargs["temperature"] == 0.1


def test_ungrounded_mode_hardens_system_prompt() -> None:
    """When RAG_ALLOW_UNGROUNDED=on and no context, system prompt must forbid synthesis."""
    captured: dict = {}

    def fake_call(messages, **_kwargs):
        captured["messages"] = messages
        return "I don't have OpenTrace data for this question.\n\nACF: no evidence."

    with mock.patch.dict(os.environ, {"RAG_ALLOW_UNGROUNDED": "on"}):
        with mock.patch("ml.rag.chatbot.generator._call_llama", side_effect=fake_call):
            generate("Explain drought resilience in Africa broadly.", [])

    sys_msg = captured["messages"][0]["content"]
    assert "CRITICAL" in sys_msg
    assert "empty" in sys_msg.lower()
    assert "Do NOT synthesise" in sys_msg
    assert "ACF: no evidence" in sys_msg


# ---------------------------------------------------------------------------
# Sprint 1 (Jul 2026) — source purity (geo conflict drop) + min-context threshold
# ---------------------------------------------------------------------------


def _geo_item(country: str, content: str = "some agricultural content") -> dict:
    return {
        "content": content,
        "_context_kind": "news",
        "metadata": {"doc_kind": "news_article", "country": country, "title": f"{country} note"},
    }


def test_drop_geo_conflicting_removes_other_country() -> None:
    """A Kenya-tagged chunk should be dropped when the query targets Senegal."""
    items = [_geo_item("Senegal"), _geo_item("Kenya")]
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
        kept = _drop_geo_conflicting(items, {"countries": ["Senegal"]})
    kept_countries = [it["metadata"]["country"] for it in kept]
    assert "Senegal" in kept_countries
    assert "Kenya" not in kept_countries


def test_drop_geo_conflicting_keeps_chunks_without_geo_metadata() -> None:
    """Chunks with no geo signal are kept (structured/global references)."""
    no_geo = {"content": "global maize overview", "_context_kind": "academic", "metadata": {}}
    items = [no_geo, _geo_item("Kenya")]
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
        kept = _drop_geo_conflicting(items, {"countries": ["Senegal"]})
    assert no_geo in kept
    assert _geo_item("Kenya")["metadata"]["country"] not in [
        it["metadata"].get("country") for it in kept
    ]


def test_drop_geo_conflicting_noop_without_target_countries() -> None:
    """No query geography → nothing dropped."""
    items = [_geo_item("Kenya"), _geo_item("Senegal")]
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
        kept = _drop_geo_conflicting(items, {})
    assert len(kept) == 2


def test_drop_geo_conflicting_can_be_disabled() -> None:
    items = [_geo_item("Kenya")]
    with mock.patch.dict(os.environ, {"RAG_DROP_GEO_CONFLICTING_CONTEXT": "off"}):
        kept = _drop_geo_conflicting(items, {"countries": ["Senegal"]})
    assert len(kept) == 1


def test_generate_geo_filter_leaves_nothing_returns_gap_message() -> None:
    """
    Real-world case: Senegal maize query pulls only Kenya/other-tagged chunks.
    With min-usable threshold set, generate() should return the structured gap
    message instead of calling the LLM with irrelevant context.
    """
    items = [_geo_item("Kenya"), _geo_item("Nigeria")]
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "should not be called"
        with mock.patch.dict(
            os.environ, {"RAG_MIN_USABLE_CONTEXT": "0"}, clear=False
        ):
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            result = generate(
                "maize yields in Senegal 2024",
                items,
                decomposition={"countries": ["Senegal"]},
            )
    assert "I don't have OpenTrace data" in result.answer
    assert "ACF: no evidence" in result.answer
    mock_llm.assert_not_called()


def test_generate_min_usable_threshold_blocks_thin_context() -> None:
    """One weak chunk + RAG_MIN_USABLE_CONTEXT=1 → structured gap, no LLM call."""
    items = [_geo_item("Senegal")]
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "should not be called"
        with mock.patch.dict(os.environ, {"RAG_MIN_USABLE_CONTEXT": "1"}, clear=False):
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            result = generate(
                "rice production in Senegal",
                items,
                decomposition={"countries": ["Senegal"]},
            )
    assert "I don't have OpenTrace data" in result.answer
    mock_llm.assert_not_called()


def test_generate_min_usable_threshold_default_allows_generation() -> None:
    """Default threshold (0) preserves prior behaviour: usable context reaches the LLM."""
    items = [_geo_item("Senegal")]
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "Rice production is strong in Senegal."
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_MIN_USABLE_CONTEXT", None)
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            result = generate(
                "rice production in Senegal",
                items,
                decomposition={"countries": ["Senegal"]},
            )
    assert "I don't have OpenTrace data" not in result.answer
    mock_llm.assert_called_once()


# ---------------------------------------------------------------------------
# Sprint 1, Week 3 — direct-answer-first (preamble stripping backstop)
# ---------------------------------------------------------------------------


def test_strip_preamble_based_on_the_context() -> None:
    out = _strip_preamble_openers(
        "Based on the context, maize yields in Kenya rose 12% in 2023."
    )
    assert out == "Maize yields in Kenya rose 12% in 2023."


def test_strip_preamble_according_to_the_context() -> None:
    out = _strip_preamble_openers(
        "According to the context, rice prices climbed sharply."
    )
    assert out == "Rice prices climbed sharply."


def test_strip_preamble_it_is_important_to_note() -> None:
    out = _strip_preamble_openers(
        "It is important to note that fertiliser costs doubled in West Africa."
    )
    assert out == "Fertiliser costs doubled in West Africa."


def test_strip_preamble_the_context_shows() -> None:
    out = _strip_preamble_openers("The context shows that drought reduced output.")
    assert out == "Drought reduced output."


def test_strip_preamble_unfortunately() -> None:
    out = _strip_preamble_openers("Unfortunately, coverage for Chad is limited.")
    assert out == "Coverage for Chad is limited."


def test_strip_preamble_unwinds_stacked_openers() -> None:
    out = _strip_preamble_openers(
        "Based on the context, it is worth noting that prices rose."
    )
    assert out == "Prices rose."


def test_strip_preamble_leaves_direct_answer_untouched() -> None:
    text = "Maize yields in Kenya rose 12% in 2023 [1]."
    assert _strip_preamble_openers(text) == text


def test_strip_preamble_does_not_touch_content_opening() -> None:
    # "This study examines" is a content-bearing opener we must NOT strip mid-answer;
    # the prompt discourages it, but if it carries meaning we leave it alone here.
    text = "Rice output grew while input costs also rose."
    assert _strip_preamble_openers(text) == text


def test_strip_preamble_never_returns_empty() -> None:
    # If the preamble is the entire content, keep the original rather than emptying.
    text = "Based on the context,"
    out = _strip_preamble_openers(text)
    assert out.strip() != ""


def test_strip_preamble_empty_input() -> None:
    assert _strip_preamble_openers("") == ""


def test_clean_answer_strips_preamble() -> None:
    out = _clean_answer("Based on the context, maize yields rose 12%.")
    assert out == "Maize yields rose 12%."


def test_generate_strips_preamble_from_llm_output() -> None:
    """End-to-end: a preamble-opening LLM answer is cleaned before returning."""
    items = [_news_item()]
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "Based on the context, rice policy shifted in Senegal.[1]"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            os.environ.pop("RAG_MIN_USABLE_CONTEXT", None)
            result = generate("What changed in rice policy?", items)
    assert not result.answer.lower().startswith("based on the context")
    assert "Rice policy shifted in Senegal." in result.answer
    assert "[1]" not in result.answer
    assert len(result.citations) == 1


def test_generate_keeps_inline_footnotes_when_requested() -> None:
    items = [_news_item()]
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "Rice policy shifted.[1]"
        result = generate(
            "Summarize with footnotes",
            items,
            export_intent="pdf",
        )
    assert "[1]" in result.answer
    assert len(result.citations) == 1


def test_build_prompt_structured_bq_unavailable_guard() -> None:
    messages = _build_prompt(
        "Which country in Africa had the highest production in 2020?",
        context_block="[News] policy chunk",
        structured_bq_unavailable=True,
    )
    sys_msg = messages[0]["content"]
    assert "invent specific production totals" in sys_msg.lower()
    assert "note for this turn" in sys_msg.lower()
    assert "bigquery structured data was attempted" not in sys_msg.lower()
    assert "opentrace structured data is unavailable" not in sys_msg.lower()
    assert "no usable rows" not in sys_msg.lower()
    assert "never mention bigquery" in sys_msg.lower()


def test_build_prompt_analytical_brief_mode_headings() -> None:
    messages = _build_prompt(
        "West Africa maize and rice report 2022",
        context_block="[News] policy chunk",
        analytical_mode=True,
    )
    sys_msg = messages[0]["content"]
    assert "ANALYTICAL BRIEF MODE" in sys_msg
    assert "ANALYTICAL BRIEFING MODE" not in sys_msg
    assert "ANALYTICAL REPORT MODE" not in sys_msg
    assert "## Key Findings" in sys_msg
    assert "## Regional & Country Picture" in sys_msg
    assert "## Production, Trade & Markets" in sys_msg
    assert "## Drivers & Context" in sys_msg
    assert "## Data Notes" in sys_msg
    assert "## Data gaps" not in sys_msg
    assert "2–4 short paragraphs" not in sys_msg
    assert "brief limits or gaps" not in sys_msg


def test_build_prompt_chat_keeps_short_paragraph_length_rule() -> None:
    messages = _build_prompt(
        "What is Ask ADZA?",
        context_block="[News] policy chunk",
        task_mode="chat",
    )
    sys_msg = messages[0]["content"]
    assert "2–4 short paragraphs" in sys_msg
    assert "ANALYTICAL BRIEF MODE" not in sys_msg


def test_build_prompt_fact_lookup_replaces_length_rule() -> None:
    messages = _build_prompt(
        "How much maize did Nigeria produce in 2020?",
        context_block="[Structured data] production row",
        task_mode="fact_lookup",
    )
    sys_msg = messages[0]["content"]
    assert "FACT LOOKUP MODE" in sys_msg
    assert "2–4 short paragraphs" not in sys_msg
    assert "brief limits or gaps" not in sys_msg


def test_citation_strip_preserves_markdown_newlines() -> None:
    registry = [
        SourceRef(source_id=1, item={"content": "a"}, citation_line="News A"),
    ]
    text = (
        "## Executive summary\nWest Africa grew.[1]\n\n"
        "## Regional overview\nMaize held up."
    )
    out = _strip_invalid_citation_markers(text, registry)
    assert "\n## Regional overview\n" in out


def test_build_prompt_africa_scope_line() -> None:
    messages = _build_prompt(
        "which country has the best agricultural activity in 2020",
        context_block="[Web] something",
    )
    sys_msg = messages[0]["content"].lower()
    assert "african agriculture" in sys_msg
    assert "non-african country" in sys_msg


def test_generate_structured_bq_unavailable_injects_guard() -> None:
    captured: dict = {}

    def fake_call(messages, **_kwargs):
        captured["messages"] = messages
        return "Structured data is unavailable; policy context only.[1]"

    items = [_news_item()]
    with mock.patch("ml.rag.chatbot.generator._call_llama", side_effect=fake_call):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            os.environ.pop("RAG_MIN_USABLE_CONTEXT", None)
            generate(
                "What does the policy brief say about millet?",
                items,
                structured_bq_unavailable=True,
            )
    sys_msg = captured["messages"][0]["content"]
    assert "CRITICAL" not in sys_msg
    assert "no usable rows" not in sys_msg.lower()


def test_generate_ranking_uses_narrative_when_bq_unavailable() -> None:
    captured: dict = {}

    def fake_call(messages, **_kwargs):
        captured["messages"] = messages
        return "From policy evidence, Kenya leads on several indicators.[1]"

    items = [_news_item()]
    with mock.patch("ml.rag.chatbot.generator._call_llama", side_effect=fake_call):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            os.environ.pop("RAG_MIN_USABLE_CONTEXT", None)
            result = generate(
                "Which country in Africa had the highest agricultural production in 2020?",
                items,
                structured_bq_unavailable=True,
            )
    assert "I don't have OpenTrace data" not in result.answer
    assert captured.get("messages")
    sys_msg = captured["messages"][0]["content"].lower()
    assert "invent specific production totals" in sys_msg
    assert "bigquery structured data was attempted" not in sys_msg
    assert "opentrace structured data is unavailable" not in sys_msg
    assert "no usable rows" not in sys_msg


def test_is_ranking_numeric_query() -> None:
    from ml.rag.chatbot.generator import is_ranking_numeric_query

    assert is_ranking_numeric_query(
        "which country in africa had the highest agricultural production in 2020"
    )
    assert not is_ranking_numeric_query("What does the policy brief say about millet?")


def test_is_numeric_data_query() -> None:
    from ml.rag.chatbot.generator import is_numeric_data_query

    assert is_numeric_data_query("how much maize did Nigeria produce in 2020")
    assert is_numeric_data_query(
        "which country in africa had the highest agricultural production in 2020"
    )
    assert not is_numeric_data_query("What does the policy brief say about millet?")


def test_generate_numeric_hard_blocks_when_bq_and_context_empty() -> None:
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "should not be called"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            os.environ.pop("RAG_MIN_USABLE_CONTEXT", None)
            result = generate(
                "how much maize did Nigeria produce in 2020?",
                [],
                structured_bq_unavailable=True,
            )
    assert "I don't have OpenTrace data" in result.answer
    mock_llm.assert_not_called()


def test_generate_numeric_uses_narrative_when_bq_unavailable() -> None:
    captured: dict = {}

    def fake_call(messages, **_kwargs):
        captured["messages"] = messages
        return "Narrative context does not include a precise production total.[1]"

    items = [_news_item()]
    with mock.patch("ml.rag.chatbot.generator._call_llama", side_effect=fake_call):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            os.environ.pop("RAG_MIN_USABLE_CONTEXT", None)
            result = generate(
                "how much maize did Nigeria produce in 2020?",
                items,
                structured_bq_unavailable=True,
            )
    assert "I don't have OpenTrace data" not in result.answer
    assert captured.get("messages")


def test_pin_bq_context_first() -> None:
    from ml.rag.chatbot.generator import pin_bq_context_first

    items = [
        {"_context_kind": "news", "content": "news"},
        {"_context_kind": "bigquery", "content": "bq"},
        {"_context_kind": "policy", "content": "policy"},
    ]
    out = pin_bq_context_first(items)
    assert [x["_context_kind"] for x in out] == ["bigquery", "news", "policy"]


def test_prefer_in_window_narrative_historical_analytical() -> None:
    from ml.rag.chatbot.generator import prefer_in_window_narrative

    items = [
        {"_context_kind": "bigquery", "content": "bq", "metadata": {}},
        {
            "_context_kind": "news",
            "content": "2026 news",
            "metadata": {"published_at": "2026-01-15", "title": "now"},
        },
        {
            "_context_kind": "news",
            "content": "2022 news",
            "metadata": {"published_at": "2022-06-01", "title": "then"},
        },
    ]
    out = prefer_in_window_narrative(
        items,
        {"time_start": "2022-01-01", "time_end": "2022-12-31"},
        analytical=True,
    )
    assert [x["content"] for x in out] == ["bq", "2022 news", "2026 news"]


def test_prefer_in_window_narrative_skips_open_ended_window() -> None:
    from ml.rag.chatbot.generator import prefer_in_window_narrative

    items = [
        {
            "_context_kind": "news",
            "content": "2026 news",
            "metadata": {"published_at": "2026-01-15"},
        },
        {
            "_context_kind": "news",
            "content": "2022 news",
            "metadata": {"published_at": "2022-06-01"},
        },
    ]
    out = prefer_in_window_narrative(
        items,
        {"time_start": "2015-01-01", "time_end": "2026-12-31"},
        analytical=True,
    )
    assert [x["content"] for x in out] == ["2026 news", "2022 news"]


def test_prefer_in_window_narrative_skips_non_analytical() -> None:
    from ml.rag.chatbot.generator import prefer_in_window_narrative

    items = [
        {
            "_context_kind": "news",
            "content": "2026 news",
            "metadata": {"published_at": "2026-01-15"},
        },
        {
            "_context_kind": "news",
            "content": "2022 news",
            "metadata": {"published_at": "2022-06-01"},
        },
    ]
    out = prefer_in_window_narrative(
        items,
        {"time_start": "2022-01-01", "time_end": "2022-12-31"},
        analytical=False,
    )
    assert [x["content"] for x in out] == ["2026 news", "2022 news"]


def test_is_comparative_bq_query() -> None:
    from ml.rag.chatbot.generator import is_comparative_bq_query

    assert is_comparative_bq_query(
        "what drove millet policy in Senegal?",
        {"intent": "diagnostic"},
    )
    assert not is_comparative_bq_query("What does the policy brief say about millet?")


def test_should_elevate_bq_context_for_comparative() -> None:
    from ml.rag.chatbot.generator import pin_bq_context_first, should_elevate_bq_context

    query = "what drove millet policy in Senegal?"
    dec = {"intent": "diagnostic"}
    assert should_elevate_bq_context(query, dec, usable_bq=True)
    items = [
        {"_context_kind": "policy", "content": "policy"},
        {"_context_kind": "bigquery", "content": "bq"},
    ]
    out = pin_bq_context_first(items)
    assert out[0]["_context_kind"] == "bigquery"


def test_generate_diagnostic_no_hard_block_when_bq_unavailable() -> None:
    captured: dict = {}

    def fake_call(messages, **_kwargs):
        captured["messages"] = messages
        return "Policy drivers include subsidies.[1]"

    items = [_news_item()]
    with mock.patch("ml.rag.chatbot.generator._call_llama", side_effect=fake_call):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_DROP_GEO_CONFLICTING_CONTEXT", None)
            os.environ.pop("RAG_MIN_USABLE_CONTEXT", None)
            result = generate(
                "what drove millet policy in Senegal?",
                items,
                decomposition={"intent": "diagnostic"},
                structured_bq_unavailable=True,
            )
    assert "Policy drivers include subsidies." in result.answer
    assert "CRITICAL" not in captured["messages"][0]["content"]


def test_build_prompt_structured_bq_unavailable_skips_guard_for_qualitative() -> None:
    messages = _build_prompt(
        "What does the policy brief say about millet?",
        context_block="[News] policy chunk",
        structured_bq_unavailable=True,
    )
    sys_msg = messages[0]["content"]
    assert "CRITICAL" not in sys_msg
    assert "no usable rows" not in sys_msg.lower()


def test_finalize_generation_result_emits_citations_span_metadata() -> None:
    registry = [
        SourceRef(
            source_id=1,
            item={
                "_context_kind": "news",
                "content": "a",
                "metadata": {
                    "doc_kind": "news_article",
                    "geo_scope": "country",
                    "geo_countries": "Kenya",
                    "published_at": "2025-06-01",
                    "document_id": "n1",
                },
            },
            citation_line="News A",
        ),
        SourceRef(
            source_id=2,
            item={"_context_kind": "bigquery", "content": "b", "metadata": {}},
            citation_line="BQ B",
        ),
    ]
    updates: list[dict] = []

    with mock.patch(
        "ml.rag.chatbot.generator.update_current_span_metadata",
        side_effect=lambda meta: updates.append(dict(meta)),
    ):
        with mock.patch("ml.rag.chatbot.generator.observed_span") as mock_span:
            mock_span.return_value.__enter__ = mock.Mock(return_value=None)
            mock_span.return_value.__exit__ = mock.Mock(return_value=False)
            result = _finalize_generation_result(
                "Answer cites [1] only.",
                registry,
                query="Kenya maize yields?",
                inline_citations=True,
            )

    assert isinstance(result, GenerationResult)
    assert len(result.citations) == 1
    assert result.citations[0]["id"] == 1
    assert result.acf is not None
    mock_span.assert_called_once()
    assert mock_span.call_args.args[0] == "citations"
    assert updates
    meta = updates[-1]
    assert meta["citation_count"] == 1
    assert meta["registry_size"] == 2
    assert meta["cited_ids"] == [1]
    assert meta["acf_status"] == "scored"
    assert "acf_score" in meta
    assert "latency_ms" in meta


def test_finalize_generation_strips_invalid_citation_markers() -> None:
    registry = [
        SourceRef(
            source_id=1,
            item={
                "_context_kind": "bigquery",
                "content": "ranked data",
                "metadata": {
                    "bq_enrichment": "ranked_table",
                    "as_of_date": "2020-01-01",
                    "geo_countries": "Nigeria",
                },
            },
            citation_line="OpenTrace FAOSTAT ranking",
        ),
    ]
    with mock.patch("ml.rag.chatbot.generator.observed_span") as mock_span:
        mock_span.return_value.__enter__ = mock.Mock(return_value=None)
        mock_span.return_value.__exit__ = mock.Mock(return_value=False)
        result = _finalize_generation_result(
            "Nigeria leads African production in 2020 [1][6].",
            registry,
            query="best agricultural activity 2020",
            inline_citations=True,
        )
    assert "[1]" in result.answer
    assert "[6]" not in result.answer
    assert len(result.citations) == 1


def test_finalize_inline_off_strips_markers_and_returns_all_citations() -> None:
    registry = [
        SourceRef(
            source_id=1,
            item={"_context_kind": "news", "content": "a", "metadata": {"doc_kind": "news_article"}},
            citation_line="News A",
        ),
        SourceRef(
            source_id=2,
            item={"_context_kind": "bigquery", "content": "b", "metadata": {}},
            citation_line="BQ B",
        ),
    ]
    with mock.patch("ml.rag.chatbot.generator.observed_span") as mock_span:
        mock_span.return_value.__enter__ = mock.Mock(return_value=None)
        mock_span.return_value.__exit__ = mock.Mock(return_value=False)
        result = _finalize_generation_result(
            "Maize yields rose in Kenya.[1] See also [2].",
            registry,
            query="Kenya maize yields?",
            inline_citations=False,
        )
    assert "[1]" not in result.answer
    assert "[2]" not in result.answer
    assert "Maize yields rose in Kenya" in result.answer
    assert len(result.citations) == 2
    assert {c["id"] for c in result.citations} == {1, 2}


def test_build_prompt_default_disables_inline_footnotes() -> None:
    messages = _build_prompt("What are maize yields in Kenya?", context_block="[Source 1] …")
    sys_msg = messages[0]["content"]
    assert "Do NOT insert [N]" in sys_msg
    assert "then add the matching footnote number [N]" not in sys_msg


def test_build_prompt_inline_citations_enabled() -> None:
    messages = _build_prompt(
        "Write a PDF report with footnotes",
        context_block="[Source 1] …",
        inline_citations=True,
    )
    sys_msg = messages[0]["content"]
    assert "then add the matching footnote number [N]" in sys_msg
    assert "Do NOT insert [N]" not in sys_msg


def _ghana_rice_bq_item() -> dict:
    return {
        "content": (
            "Ghana produced 973,000 metric tons of rice in 2020 "
            "(OpenTrace structured production data)."
        ),
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "source_id": "stg_faostat_production:country_name=Ghana:year=2020",
            "country_name": "Ghana",
            "product_name": "Rice",
            "element": "Production",
            "year": 2020,
            "value": 973000,
            "unit": "t",
            "direction": "unknown",
            "geo_country_primary": "Ghana",
            "geo_countries": "Ghana",
            "as_of_date": "2020-01-01",
            "tier": 2,
            "sql": (
                "SELECT country_name, product_name, element, year, value "
                "FROM `proj.staging_dev.stg_faostat_production` "
                "WHERE country_name = 'Ghana' AND year = 2020"
            ),
        },
    }


def test_dedupe_bq_context_items_collapses_same_source_id() -> None:
    item = _ghana_rice_bq_item()
    dupes = [dict(item) for _ in range(7)]
    deduped = dedupe_bq_context_items(dupes + [_news_item()])
    assert len(deduped) == 2
    assert sum(1 for x in deduped if x.get("_context_kind") == "bigquery") == 1


def test_build_context_block_registers_bq_when_body_truncated() -> None:
    """Citations must survive tight budgets that leave no room for chunk body."""
    items = [_ghana_rice_bq_item()]
    block, registry = _build_context_block(items, budget=120, chunk_cap=120)
    assert len(registry) == 1
    assert registry[0].citation_line is not None
    assert "country_name=Ghana" in registry[0].citation_line
    assert block == "" or "[Source 1]" not in block or len(block) <= 120


def test_registry_from_context_items_builds_citable_refs() -> None:
    items = [_ghana_rice_bq_item(), _news_item()]
    registry = _registry_from_context_items(items)
    assert len(registry) == 2
    assert registry[0].source_id == 1
    assert registry[1].source_id == 2


def test_ghana_rice_pack_finalize_non_empty_citations_and_acf() -> None:
    items = [_ghana_rice_bq_item()]
    _, registry = _build_context_block(items, budget=120, chunk_cap=120)
    if not registry:
        registry = _registry_from_context_items(items)
    with mock.patch("ml.rag.chatbot.generator.observed_span") as mock_span:
        mock_span.return_value.__enter__ = mock.Mock(return_value=None)
        mock_span.return_value.__exit__ = mock.Mock(return_value=False)
        result = _finalize_generation_result(
            "Ghana produced 973,000 metric tons of rice in 2020.",
            registry,
            query="what was the production of rice like in ghana in 2020",
            inline_citations=False,
        )
    assert len(result.citations) == 1
    assert result.citations[0]["kind"] == "structured_data"
    assert result.acf is not None
    assert result.acf.band != "no_evidence"
    assert result.acf.claim_level is not None


def test_normalize_multi_source_brackets() -> None:
    out = _normalize_inline_citations("Trends rose [Source 1, 3] across regions.")
    assert "[1]" in out
    assert "[3]" in out
    assert "[Source 1, 3]" not in out


def test_normalize_strips_wikipedia_pipe_label() -> None:
    out = _normalize_inline_citations("Maize is a staple [Wikipedia | Maize] in West Africa.")
    assert "[Wikipedia" not in out
    assert "| Maize]" not in out


def test_normalize_collapses_duplicate_markers() -> None:
    out = _normalize_inline_citations("Prices rose ([3]) [3] sharply.")
    assert out.count("[3]") == 1


def test_strip_invalid_citation_markers_drops_unknown_id() -> None:
    registry = [SourceRef(i, _news_item(), f"[News] {i}") for i in range(1, 6)]
    out = _strip_invalid_citation_markers("Trends rose [1] and [99] here.", registry)
    assert "[1]" in out
    assert "[99]" not in out


def test_format_academic_citation_strips_affiliation_digit() -> None:
    cite = format_academic_citation(
        {
            "authors": "Nicolas Depetris Chauvin 1, Francis Mulangu and Guido Porto 1",
            "publication_year": "2012",
            "article_title": "When Africa awakens",
        }
    )
    assert "Chauvin 1" not in cite
    assert "Porto 1" not in cite
    assert "2012" in cite


def test_format_academic_citation_no_leading_comma() -> None:
    cite = format_academic_citation(
        {
            "authors": ", Elizabeth J. Z. Robinson 1",
            "publication_year": "2021",
            "article_title": "Forest conservation",
        }
    )
    assert not cite.startswith(",")
    assert "Robinson" in cite


def test_format_source_citation_rejects_body_prose() -> None:
    item = {
        "_context_kind": "academic",
        "metadata": {
            "authors": "Someone",
            "article_title": "x" * 400,
            "publication_year": "2020",
        },
    }
    assert _format_source_citation(item) is None


def test_dedupe_context_items_same_doi() -> None:
    items = [
        {
            "content": "chunk one " + ("x" * 40),
            "_context_kind": "academic",
            "metadata": {"doi": "10.1234/abc", "article_title": "Paper A", "authors": "Smith"},
        },
        {
            "content": "chunk two " + ("y" * 40),
            "_context_kind": "academic",
            "metadata": {"doi": "https://doi.org/10.1234/abc", "article_title": "Paper A", "authors": "Smith"},
        },
    ]
    out = dedupe_context_items(items)
    assert len(out) == 1


def test_clean_answer_splits_midline_heading() -> None:
    out = _clean_answer("Intro text ## Key Findings Maize production rose in 2020.")
    assert "\n\n## Key Findings\n\n" in out
    assert "Maize production rose" in out


def test_finalize_inline_on_returns_registry_without_markers() -> None:
    registry = [
        SourceRef(1, _news_item(), "[News] Title A"),
        SourceRef(2, _bq_item(), "[Structured data] BQ"),
    ]
    with mock.patch("ml.rag.chatbot.generator.observed_span") as mock_span:
        mock_span.return_value.__enter__ = mock.Mock(return_value=None)
        mock_span.return_value.__exit__ = mock.Mock(return_value=False)
        result = _finalize_generation_result(
            "Andersson Djurfeldt et al. (2018) note rising yields.",
            registry,
            query="West Africa maize",
            inline_citations=True,
        )
    assert len(result.citations) == 2
    assert "[1]" not in result.answer


def test_finalize_inline_on_referenced_subset_when_markers_present() -> None:
    registry = [
        SourceRef(1, _news_item(), "[News] Title A"),
        SourceRef(2, _bq_item(), "[Structured data] BQ"),
    ]
    with mock.patch("ml.rag.chatbot.generator.observed_span") as mock_span:
        mock_span.return_value.__enter__ = mock.Mock(return_value=None)
        mock_span.return_value.__exit__ = mock.Mock(return_value=False)
        result = _finalize_generation_result(
            "Yields rose.[1]",
            registry,
            query="Kenya maize",
            inline_citations=True,
        )
    assert len(result.citations) == 1
    assert result.citations[0]["id"] == 1


def test_citation_url_prefers_doi() -> None:
    url = _citation_url(
        "academic",
        {"doi": "10.1234/abc", "url": "https://example.com/paper"},
    )
    assert url == "https://doi.org/10.1234/abc"


def test_citation_url_deprioritizes_social_share() -> None:
    url = _citation_url(
        "news",
        {
            "url": "https://www.linkedin.com/posts/example-maize",
            "canonical_url": "https://www.weforum.org/agenda/2020/maize/",
        },
    )
    assert url == "https://www.weforum.org/agenda/2020/maize/"


def test_citation_url_keeps_social_when_only_option() -> None:
    url = _citation_url(
        "news",
        {"url": "https://www.linkedin.com/posts/example-maize"},
    )
    assert "linkedin.com" in (url or "")

