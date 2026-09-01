"""Tests for entity → multi-measure retrieval contracts."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.agri_measure_ontology import resolve_measure, resolve_measures
from ml.rag.chatbot.corpus_catalog import select_corpora
from ml.rag.chatbot.facet_enrich import enrich_decomposition_facets
from ml.rag.chatbot.bq_table_schema_yaml import list_mart_table_index
from ml.rag.chatbot.retrieval_contract import build_retrieval_contract
from ml.rag.chatbot.task_mode import needs_clarify, resolve_task_mode


_KNOWN = {r["table_id"] for r in list_mart_table_index()}


def test_resolve_measures_maize_kenya_yield() -> None:
    q = "What was maize yield in Kenya in 2020?"
    dec = {
        "geography": ["Kenya"],
        "entities": ["maize", "yield"],
        "domains": ["agriculture"],
        "time_end": "2020-12-31",
    }
    hits = resolve_measures(q, dec, top_k=3)
    ids = [h.measure.id for h in hits]
    assert "yield" in ids or "production" in ids
    assert resolve_measure(q, dec) is not None


def test_contract_maize_kenya_includes_production_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SQL_COMPILER", "0")
    q = "What was maize yield in Kenya in 2020?"
    dec = {
        "geography": ["Kenya"],
        "entities": ["maize", "yield"],
        "domains": ["Agricultural Production & Yield"],
        "time_end": "2020-12-31",
    }
    contract = build_retrieval_contract(q, decomposition=dec, known_tables=_KNOWN)
    assert "yield" in contract.primary_measures or "production" in contract.primary_measures
    assert "fct_yield" in contract.bq_tables or "fct_production" in contract.bq_tables
    assert any("Production" in t or "production" in t.lower() for t in contract.corpus_domain_tags) or any(
        "Yield" in t or "Production" in t for t in contract.corpus_domain_tags
    )
    assert not contract.skip_bq


def test_contract_compiler_path_skips_bq_intents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SQL_COMPILER", "1")
    q = "What was maize yield in Kenya in 2020?"
    dec = {
        "geography": ["Kenya"],
        "entities": ["maize", "yield"],
        "domains": ["Agricultural Production & Yield"],
        "time_end": "2020-12-31",
    }
    contract = build_retrieval_contract(q, decomposition=dec, known_tables=_KNOWN)
    assert contract.primary_measures
    assert contract.bq_tables == []
    assert contract.bq_intents == []


def test_contract_sahel_food_security_multi_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SQL_COMPILER", "0")
    q = "Assess food security risk across the Sahel"
    dec = {
        "geography": ["Mali", "Niger", "Burkina Faso"],
        "entities": ["food security", "Sahel"],
        "expanded_regions": ["sahel"],
        "domains": [],
        "time_start": "2020-01-01",
        "time_end": "2025-12-31",
    }
    contract = build_retrieval_contract(q, decomposition=dec, known_tables=_KNOWN)
    assert "food_security_ipc" in contract.primary_measures
    assert "fct_food_security" in contract.bq_tables
    assert "fct_production" in contract.bq_tables
    assert "fct_research_expenditure" not in contract.bq_tables
    assert len(contract.bq_intents) >= 2


def test_contract_nigeria_agribusiness_not_forced_food_security() -> None:
    q = "what do you think about agribusiness in nigeria now"
    dec = {
        "geography": ["Nigeria"],
        "entities": ["agribusiness", "Nigeria"],
        "intent": "descriptive",
        "domains": [],
    }
    enriched = enrich_decomposition_facets(q, dec)
    assert "agribusiness" in [e.lower() for e in enriched["entities"]]
    contract = build_retrieval_contract(q, decomposition=enriched, known_tables=_KNOWN)
    assert "food_security_ipc" not in contract.primary_measures
    assert not needs_clarify(q, enriched)
    assert resolve_task_mode(q, enriched) != "clarify"
    selection = select_corpora(
        enriched,
        query=q,
        task_mode="chat",
        corpus_domain_tags=contract.corpus_domain_tags,
    )
    assert "news" in selection.active
    assert len(selection.active) >= 3


def test_enrich_ipc_grounds_food_security_entity() -> None:
    out = enrich_decomposition_facets(
        "What is the IPC situation in Somalia?",
        {"geography": ["Somalia"], "entities": ["IPC"], "domains": []},
    )
    ent = [e.lower() for e in out["entities"]]
    assert "food security" in ent
