"""Decomposition facets must be grounded in the user query text."""

from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.geo_iso3 import name_to_iso3

from ml.rag.chatbot.query_decomposer import (
    decompose_query,
    facet_grounded_in_query,
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
        "ml.rag.chatbot.query_decomposer._decompose_backend_configured",
        return_value=True,
    ):
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


def test_accented_country_with_curly_apostrophe_is_not_lost() -> None:
    query = "Compare rice imports in Senegal, Côte d’Ivoire and Madagascar from 2015 to 2024."
    out = decompose_query(query, use_llm=False)
    assert {name_to_iso3(place) for place in out["geography"]} == {"SEN", "CIV", "MDG"}


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
        "ml.rag.chatbot.query_decomposer._decompose_backend_configured",
        return_value=True,
    ):
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


def test_facet_grounded_via_language_enrich_alt() -> None:
    """English enrich can evidence a country missing from the local-language surface form."""
    original = "kedu ihe mere ka mmeputa akpu gbawe kemgbe afo ise"
    enrich = "What caused cassava production in Nigeria to change over five years"
    assert facet_grounded_in_query("Nigeria", original) is False
    assert facet_grounded_in_query("Nigeria", original, enrich) is True


def test_facet_grounded_in_query_public_wrapper_rejects_unrelated_country() -> None:
    """Regression: Sprint 2 'Geography: France'/'Geography: Netherlands' routing bug.

    A country that only exists in stale conversation context (or any source
    other than the raw current-turn query) must never be treated as grounded,
    even when it is a real, well-known country name. This backs the defensive
    re-grounding guard applied to reasoner-plan ``geos`` in
    ``graph.py::node_decompose`` before they overwrite ``dec["geography"]``.
    """
    q = "What is the coffee yield from Nigeria over the last year"
    assert facet_grounded_in_query("Nigeria", q) is True
    assert facet_grounded_in_query("Netherlands", q) is False
    assert facet_grounded_in_query("France", q) is False


def test_reasoner_geos_reground_drops_ungrounded_country() -> None:
    """Simulates the exact graph.py list-comprehension guard against a
    reasoner plan carrying a stale/unrelated country from a prior turn.
    """
    q = "What is the coffee yield from Nigeria over the last year"
    stale_reasoner_geos = ("Netherlands",)
    grounded = [g for g in stale_reasoner_geos if facet_grounded_in_query(g, q)]
    assert grounded == []

    correct_reasoner_geos = ("Nigeria",)
    grounded = [g for g in correct_reasoner_geos if facet_grounded_in_query(g, q)]
    assert grounded == ["Nigeria"]
