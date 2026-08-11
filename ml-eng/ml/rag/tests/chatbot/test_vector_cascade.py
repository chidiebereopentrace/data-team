"""Tests for vector cascade widen-before-drop and geo post-filter wiring."""
from __future__ import annotations

from unittest.mock import MagicMock

from ml.rag.chatbot.graph import (
    _cascade_attempt_label,
    _news_kwargs,
    _post_filter_geography,
    _retrieve_vector_cascade,
    _widen_time_kwargs,
)


def test_widen_time_kwargs_expands_year() -> None:
    base = {
        "published_at_from": "2020-01-01",
        "published_at_to": "2020-12-31",
        "geo_country": "Kenya",
        "top_k": 5,
    }
    widened = _widen_time_kwargs(base)
    assert widened is not None
    assert widened["published_at_from"] == "2019-01-01"
    assert widened["published_at_to"] == "2021-12-31"
    assert widened["geo_country"] == "Kenya"


def test_cascade_attempt_label_time_widen() -> None:
    base = {"published_at_from": "2020-01-01", "published_at_to": "2020-12-31", "geo_country": "Kenya"}
    wide = {"published_at_from": "2019-01-01", "published_at_to": "2021-12-31", "geo_country": "Kenya"}
    assert _cascade_attempt_label(base, wide) == "time_widen"
    assert _cascade_attempt_label(base, base) == "none"
    no_time = {"geo_country": "Kenya"}
    assert _cascade_attempt_label(base, no_time) == "no_time"


def test_cascade_tries_time_widen_before_drop(monkeypatch) -> None:
    monkeypatch.setenv("RAG_NEWS_GEO_FALLBACK", "on")
    monkeypatch.setenv("RAG_NEWS_TIME_FALLBACK", "on")
    calls: list[dict] = []

    def _fake_retrieve(query: str, **kwargs):
        calls.append(dict(kwargs))
        # Only succeed on time-widened attempt (year 2019/2021).
        ts = str(kwargs.get("published_at_from") or "")
        if ts.startswith("2019"):
            return [{"content": "hit", "score": 0.9, "metadata": {}}]
        return []

    vr = MagicMock()
    vr.collection_name = "news_data"
    vr.retrieve.side_effect = _fake_retrieve

    base = {
        "top_k": 5,
        "doc_kind": "news_article",
        "geo_country": "Kenya",
        "published_at_from": "2020-01-01",
        "published_at_to": "2020-12-31",
    }
    out = _retrieve_vector_cascade(
        vr,
        "maize Kenya 2020",
        base_kwargs=base,
        countries=["Kenya"],
        has_time=True,
        geo_fallback_env="RAG_NEWS_GEO_FALLBACK",
        time_fallback_env="RAG_NEWS_TIME_FALLBACK",
    )
    assert out
    assert out[0]["metadata"].get("constraint_relaxed") == "time_widen"
    assert len(calls) >= 2
    assert calls[0].get("published_at_from") == "2020-01-01"
    assert calls[1].get("published_at_from") == "2019-01-01"
    # Must not have dropped geo before succeeding.
    assert calls[1].get("geo_country") == "Kenya"


def test_post_filter_geography_avoids_niger_nigeria() -> None:
    items = [
        {
            "content": "Markets in Nigeria",
            "metadata": {"geo_country_primary": "Nigeria"},
        },
        {
            "content": "Sahel update",
            "metadata": {"geo_country_primary": "Niger"},
        },
    ]
    kept = _post_filter_geography(items, ["Niger"])
    assert len(kept) == 1
    assert kept[0]["metadata"]["geo_country_primary"] == "Niger"


def test_news_domain_filter_default_on(monkeypatch) -> None:
    monkeypatch.delenv("RAG_NEWS_DOMAIN_FILTER", raising=False)
    kw = _news_kwargs({}, {"domains": ["food_security", "trade"]})
    assert kw.get("domains_substring") == "food_security"


def test_news_domain_filter_can_disable(monkeypatch) -> None:
    monkeypatch.setenv("RAG_NEWS_DOMAIN_FILTER", "off")
    kw = _news_kwargs({}, {"domains": ["food_security"]})
    assert "domains_substring" not in kw
