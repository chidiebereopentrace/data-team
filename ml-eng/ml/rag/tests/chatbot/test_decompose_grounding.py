"""Decomposition facets must be grounded in the user query text."""

from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.query_decomposer import (
    decompose_query,
    normalize_geography_for_filter,
    wants_africa_default_scope,
)


def test_igbo_yam_query_does_not_keep_hallucinated_nigeria() -> None:
    q = "kedu obodo kacha ako ji na mba africa"
    fake_llm = {
        "intent": "locate",
        "entities": ["Nigeria", "Africa"],
        "geography": ["Nigeria"],
        "domains": ["agriculture"],
        "time_start": "",
        "time_end": "",
    }
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._call_llama_decompose",
        return_value=fake_llm,
    ):
        out = decompose_query(q)
    geo_l = [g.lower() for g in out.get("geography") or []]
    ent_l = [e.lower() for e in out.get("entities") or []]
    assert "nigeria" not in geo_l
    assert "nigeria" not in ent_l
    # Africa may remain as an entity (present in query) but is dropped from geo filters.
    assert "africa" in ent_l or "africa" not in geo_l


def test_geography_drops_literal_country_stopword() -> None:
    assert normalize_geography_for_filter(["country", "Nigeria", "world"]) == ["Nigeria"]


def test_which_country_agricultural_activity_africa_default() -> None:
    q = "which country has the best agricultural activity in 2020"
    fake_llm = {
        "intent": "descriptive",
        "entities": ["agricultural activity"],
        "geography": ["country"],
        "domains": ["agribusiness"],
        "time_start": "2020-01-01",
        "time_end": "2020-12-31",
    }
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._call_llama_decompose",
        return_value=fake_llm,
    ):
        out = decompose_query(q)
    assert "country" not in [g.lower() for g in out.get("geography") or []]
    assert out.get("africa_default") is True
    assert any(str(e).lower() == "africa" for e in (out.get("entities") or []))
    assert wants_africa_default_scope(q) is True


def test_named_country_skips_africa_default() -> None:
    q = "which region in Nigeria has the highest maize production in 2020"
    assert wants_africa_default_scope(q) is False
