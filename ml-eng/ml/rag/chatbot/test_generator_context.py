"""Unit tests for generator context packing and citation filtering."""

from __future__ import annotations

import os
from unittest import mock

from ml.rag.chatbot.generator import (
    SourceRef,
    _append_structured_citations,
    _build_context_block,
    _context_max_chars,
    _format_source_citation,
    _generate_max_tokens,
    extract_referenced_source_ids,
)


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
    assert "[Source 1" in block
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
    assert "yield_raw_data" in line
    assert "Senegal" in line or "rice" in line


def test_extract_referenced_source_ids() -> None:
    text = "Rice yields rose [Source 1] while policy shifted per Source 3."
    assert extract_referenced_source_ids(text) == {1, 3}


def test_append_citations_referenced_only() -> None:
    registry = [
        SourceRef(1, _news_item(), "[News] Senegal rice policy shift — AgriNews (2023)"),
        SourceRef(2, _bq_item(), "[Structured data] yield_raw_data (country=Senegal)"),
    ]
    answer = "Production trends improved [Source 1] in recent seasons."
    with mock.patch.dict(os.environ, {"RAG_CITATIONS_MODE": "referenced"}):
        out = _append_structured_citations(answer, registry)
    assert "Sources" in out
    assert "[Source 1]" in out
    assert "[Source 2]" not in out


def test_append_citations_all_mode() -> None:
    registry = [
        SourceRef(1, _news_item(), "[News] Senegal rice policy shift — AgriNews (2023)"),
        SourceRef(2, _bq_item(), "[Structured data] yield_raw_data"),
    ]
    answer = "Production trends improved."
    with mock.patch.dict(os.environ, {"RAG_CITATIONS_MODE": "all"}):
        out = _append_structured_citations(answer, registry)
    assert "[Source 1]" in out
    assert "[Source 2]" in out


def test_append_citations_no_refs_when_unreferenced() -> None:
    registry = [SourceRef(1, _news_item(), "[News] Title")]
    answer = "Production trends improved with no inline cites."
    with mock.patch.dict(os.environ, {"RAG_CITATIONS_MODE": "referenced"}):
        out = _append_structured_citations(answer, registry)
    assert "Sources" not in out
