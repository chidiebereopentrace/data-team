"""Mode compiler invariants — property-style, not phrase catalogs."""
from __future__ import annotations

from ml.rag.chatbot.continental_scope import (
    is_continental_count_query,
    wants_africa_default_scope,
)
from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.query_decomposer import apply_africa_default_scope
from ml.rag.chatbot.task_mode import needs_clarify, resolve_task_mode


def test_africa_count_query_sets_africa_default() -> None:
    q = "how many countries in africa produce yam and cassava"
    assert is_continental_count_query(q)
    assert wants_africa_default_scope(q, extract_countries=False)
    dec = apply_africa_default_scope(
        {"entities": ["yam", "cassava"], "geography": []},
        q,
    )
    assert dec.get("africa_default") is True


def test_africa_count_not_clarify() -> None:
    q = "how many countries in africa produce yam and cassava"
    dec = apply_africa_default_scope(
        {"entities": ["yam", "cassava", "production"], "geography": [], "primary_measures": ["production"]},
        q,
    )
    assert resolve_task_mode(q, dec) != "clarify"
    assert not needs_clarify(q, dec)


def test_africa_count_compiles_served_geo_grain() -> None:
    q = "how many countries in africa produce yam and cassava"
    dec = apply_africa_default_scope(
        {
            "entities": ["yam", "cassava", "production"],
            "geography": [],
            "primary_measures": ["production"],
        },
        q,
    )
    turn = compile_turn_contract(q, dec, task_mode_hint="fact_lookup")
    assert turn.serve_status != "clarify"
    assert turn.geo_grain == "africa"
    assert turn.job == "rank"
