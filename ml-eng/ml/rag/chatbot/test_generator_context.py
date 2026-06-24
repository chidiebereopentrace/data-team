"""Unit tests for generator context packing and citation filtering."""

from __future__ import annotations

import os
from unittest import mock

from ml.rag.chatbot.generator import (
    GenerationResult,
    SourceRef,
    _append_structured_citations,
    _build_context_block,
    _context_max_chars,
    _format_source_citation,
    _generate_max_tokens,
    _normalize_inline_citations,
    _strip_model_sources_appendix,
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
        assert _generate_max_tokens() == 2048
    with mock.patch.dict(os.environ, {"RAG_GENERATE_MAX_TOKENS": "4096"}):
        assert _generate_max_tokens() == 4096


def test_context_max_chars_memory_reduces_budget() -> None:
    with mock.patch.dict(os.environ, {"RAG_GENERATE_CONTEXT_MAX_CHARS": "12000"}):
        assert _context_max_chars("") == 12000
        assert _context_max_chars("x" * 2000) == 10000


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
    assert "Sources" not in out


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
    assert "[1]" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0]["id"] == 1
