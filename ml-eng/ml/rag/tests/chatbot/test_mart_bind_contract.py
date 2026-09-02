"""Property tests for TableBindContract — table × facet binding, not class-by-class."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.bq_table_schema_yaml import (
    _AFRICA_COUNTRY_ISO3,
    compile_table_bind_contract,
    geo_column,
    list_mart_table_index,
    resolve_geo_filter_values,
    resolve_geo_literals_for_table,
    year_column,
)
from ml.rag.chatbot.class_engines.prod import ProdEngine
from ml.rag.chatbot.schema_card import load_schema_card


@pytest.mark.parametrize("iso3", sorted(set(_AFRICA_COUNTRY_ISO3.values()))[:12])
def test_iso3_maps_to_country_name_on_agg_production_annual(iso3: str) -> None:
    literals = resolve_geo_literals_for_table("agg_production_annual", [iso3])
    assert literals
    assert literals[0] != iso3
    assert len(literals[0]) > 3


def test_resolve_geo_filter_values_kenya_rice_table() -> None:
    assert resolve_geo_filter_values("agg_production_annual", ["KEN"]) == ["Kenya"]
    assert resolve_geo_filter_values("agg_production_country_year", ["KEN"]) == ["KEN"]


def test_compile_table_bind_contract_has_nomenclature() -> None:
    facets = {
        "geography": ["Kenya"],
        "entities": ["Rice"],
        "time_start": "2016-01-01",
        "time_end": "2016-12-31",
        "primary_measures": ["production"],
    }
    contract = compile_table_bind_contract(
        "agg_production_country_year",
        facets=facets,
        card=load_schema_card("PROD") or {},
        query="what is the production of rice in kenya in 2016",
        country_labels=["KEN"],
    )
    assert contract.geo_column == "country_iso3"
    assert "KEN" in contract.geo_literals
    assert contract.time_column == "year"
    assert "TABLE:" in contract.nomenclature
    assert "country_iso3" in contract.nomenclature


def test_agg_production_annual_uses_time_key_not_year() -> None:
    assert year_column("agg_production_annual") == "time_key"
    contract = compile_table_bind_contract(
        "agg_production_annual",
        facets={"time_start": "2016-01-01", "time_end": "2016-12-31", "geography": ["Kenya"]},
        query="production kenya 2016",
        country_labels=["KEN"],
    )
    assert contract.time_column == "time_key"
    assert any("time_key" in (contract.time_sql or "") for _ in [0]) or contract.time_sql


@pytest.mark.parametrize(
    "table_id",
    [
        tid
        for tid in ("agg_production_country_year", "agg_production_annual", "fct_production")
        if tid in {r["table_id"] for r in list_mart_table_index()}
    ],
)
def test_geo_column_exists_in_yaml(table_id: str) -> None:
    col = geo_column(table_id)
    assert col is not None


def test_prod_engine_returns_planned_without_sql() -> None:
    engine = ProdEngine()
    result = engine.run_plan(
        "what is the production of rice in kenya in 2016",
        facets={
            "geography": ["Kenya"],
            "entities": ["Rice"],
            "time_start": "2016-01-01",
            "time_end": "2016-12-31",
            "primary_measures": ["production"],
        },
        card=load_schema_card("PROD") or {},
    )
    assert result.status == "planned"
    assert result.sql is None
    assert result.bind_contract is not None
    assert result.query_intents
