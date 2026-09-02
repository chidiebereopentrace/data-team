"""SQL compiler unit tests."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.sql_compiler import SqlRequest, assemble_sql, compile_sql, sql_compiler_enabled


def test_sql_compiler_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RAG_SQL_COMPILER", raising=False)
    assert sql_compiler_enabled() is True


def test_assemble_sql_removed() -> None:
    card = load_schema_card("FS") or {}
    req = SqlRequest(
        class_code="FS",
        table_id="fct_food_security",
        geos=["GHA"],
        year_start=2020,
        year_end=2024,
        shape="series",
    )
    with pytest.raises(NotImplementedError):
        assemble_sql(req, card)


def test_compile_delegates_to_bind_contract_path() -> None:
    card = load_schema_card("FS") or {}
    req = SqlRequest(
        class_code="FS",
        table_id="fct_food_security",
        geos=["GHA", "NGA"],
        year_start=2020,
        year_end=2024,
        shape="panel",
        value_hits={"country_iso3": ["GHA", "NGA"]},
    )
    sql, err = compile_sql(req, card)
    assert sql is None
    assert "bind_contract" in err
