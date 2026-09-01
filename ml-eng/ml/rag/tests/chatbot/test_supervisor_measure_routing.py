"""Tests for measure-primary supervisor routing."""
from __future__ import annotations

from ml.rag.chatbot.agri_measure_ontology import resolve_measures
from ml.rag.chatbot.class_supervisor import compile_supervisor_plan, corpora_for_supervisor_plan
from ml.rag.chatbot.facet_enrich import enrich_decomposition_facets
from ml.rag.chatbot.query_normalize import normalize_query_text


def _plan(q: str, *, geo: list[str] | None = None) -> tuple:
    qn = normalize_query_text(q)
    dec = enrich_decomposition_facets(qn, {"geography": geo or [], "entities": [], "domains": []})
    if geo:
        dec["geography"] = geo
    hits = resolve_measures(qn, dec)
    mh = hits[0] if hits else None
    sp = compile_supervisor_plan(qn, decomposition=dec, measure_hit=mh)
    return sp, mh


def test_nigeria_maize_price_routes_prc() -> None:
    q = "What was the retail price of maize in Nigeria in 2022?"
    sp, mh = _plan(q, geo=["Nigeria"])
    assert mh is not None
    assert mh.measure.id == "market_price"
    assert sp.classes == ("PRC",)
    assert "news" in corpora_for_supervisor_plan(sp)


def test_prize_typo_maize_price_routes_prc() -> None:
    q = "what is the prize od maize in nigeria in 2022"
    sp, mh = _plan(q, geo=["Nigeria"])
    assert mh is not None
    assert mh.measure.id == "market_price"
    assert sp.classes == ("PRC",)


def test_maize_production_routes_prod_not_prc() -> None:
    q = "What was maize production in Nigeria in 2022?"
    sp, mh = _plan(q, geo=["Nigeria"])
    assert mh is not None
    assert mh.measure.id == "production"
    assert sp.classes == ("PROD",)
