"""Tests for diversity-aware context packing across corpora."""
from __future__ import annotations

from ml.rag.chatbot.context_diversity import (
    dedupe_context_items,
    diversify_context_pack,
    normalize_context_kind,
)


def _item(kind: str, title: str, *, score: float = 0.5, url: str | None = None) -> dict:
    meta: dict = {"title": title}
    if url:
        meta["url"] = url
    return {
        "content": f"{title} body " + ("x" * 40),
        "_context_kind": kind,
        "source": kind,
        "score": score,
        "metadata": meta,
    }


def test_normalize_context_kind_aliases() -> None:
    assert normalize_context_kind({"_context_kind": "news_article"}) == "news"
    assert normalize_context_kind({"source": "bigquery"}) == "bigquery"
    assert normalize_context_kind({"_context_kind": "public_reports"}) == "public_report"


def test_dedupe_same_url() -> None:
    items = [
        _item("news", "Story A", url="https://example.com/a"),
        _item("news", "Story A copy", url="https://example.com/a"),
        _item("policy", "Policy B", url="https://example.com/b"),
    ]
    out = dedupe_context_items(items)
    assert len(out) == 2


def test_diversify_pack_mixes_corpora_not_news_flood() -> None:
    items = [
        *[_item("news", f"News {i}", score=0.9 - i * 0.01, url=f"https://n/{i}") for i in range(20)],
        _item("policy", "Policy one", score=0.4, url="https://p/1"),
        _item("policy", "Policy two", score=0.39, url="https://p/2"),
        _item("public_report", "Public one", score=0.38, url="https://r/1"),
        _item("academic", "Paper one", score=0.37, url="https://a/1"),
        _item("bigquery", "BQ rows", score=0.99),
    ]
    packed = diversify_context_pack(items, top_k=14, task_mode="chat", per_kind_min=2, bq_max=4)
    kinds = [normalize_context_kind(i) for i in packed]
    assert kinds.count("bigquery") >= 1
    assert "news" in kinds
    assert "policy" in kinds
    assert "public_report" in kinds or "academic" in kinds
    # Must not be all news despite news dominating scores.
    assert kinds.count("news") < len(packed)
    assert len(packed) <= 14


def test_diversify_pack_dedupes_before_quota() -> None:
    items = [
        _item("news", "Dup", url="https://example.com/dup"),
        _item("news", "Dup again", url="https://example.com/dup"),
        _item("policy", "Policy", url="https://example.com/pol"),
        _item("public_report", "Report", url="https://example.com/rep"),
    ]
    packed = diversify_context_pack(items, top_k=10, task_mode="briefing", per_kind_min=1)
    urls = [str((i.get("metadata") or {}).get("url") or "") for i in packed]
    assert urls.count("https://example.com/dup") == 1
