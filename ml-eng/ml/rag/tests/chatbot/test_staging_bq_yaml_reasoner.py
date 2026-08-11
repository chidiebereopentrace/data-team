"""Tests for staging_dev YAML catalog, byte budgets, and SQL reasoner."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from ml.rag.chatbot.bq_byte_budget import (
    pack_lines,
    trim_bq_result_contents,
    truncate_utf8,
    utf8_len,
)
from ml.rag.chatbot.bq_sql_reasoner import reason_bq_sql_plan
from ml.rag.chatbot.bq_table_schema_yaml import (
    format_reasoner_index,
    format_table_schema,
    list_staging_table_index,
    load_table_schema,
    pack_selected_table_hints,
)
from ml.rag.retrievers import bq_retriever
from ml.rag.retrievers.bq_retriever import BQRetriever, _get_datasets_config, _validate_sql

_YAML_DIR = Path(__file__).resolve().parents[2] / "bq_tables_yaml_files"


def test_staging_yaml_index_has_stg_yield() -> None:
    rows = list_staging_table_index()
    ids = {r["table_id"] for r in rows}
    assert "stg_yield_raw_data" in ids
    assert "stg_fews_food_security" in ids
    assert len(rows) >= 40
    assert any(r.get("rels") for r in rows)


def test_every_stg_yaml_has_semantic_relationships() -> None:
    paths = list(_YAML_DIR.glob("stg_*.yml"))
    assert len(paths) >= 40
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        rel = data.get("semantic_relationships")
        assert isinstance(rel, dict), path.name
        assert rel.get("joins_with") is not None or rel.get("companions") is not None or rel.get(
            "do_not_join"
        ), path.name


def test_format_table_schema_includes_semantic_relationships() -> None:
    text = format_table_schema("stg_yield_raw_data", max_bytes=5000)
    assert "staging_dev.stg_yield_raw_data" in text
    assert "Semantic relationships" in text
    assert "stg_faostat_production" in text or "joins_with" in text.lower()


def test_format_table_schema_filters_joins_to_selected_set() -> None:
    full = format_table_schema("stg_yield_raw_data", max_bytes=8000)
    filtered = format_table_schema(
        "stg_yield_raw_data",
        max_bytes=8000,
        selected_tables={"stg_yield_raw_data"},
    )
    assert "stg_faostat_production on=" in full
    # Relationship joins are filtered; table may still be named in SQL hints.
    assert "stg_faostat_production on=" not in filtered


def test_production_schema_pack_surfaces_product_samples_and_guidance() -> None:
    text = format_table_schema("stg_faostat_production", max_bytes=16000)
    assert "Product value samples:" in text
    assert "Wheat" in text
    assert "Millet" in text
    assert "Milk, Total" in text
    assert "Crops" in text
    assert "aggregate" in text.lower() or "totals" in text.lower()
    assert "product_name" in text
    assert "Prefer primary product_name" in text or "primary commodities" in text.lower()


def test_emissions_schema_pack_surfaces_samples_and_guidance() -> None:
    text = format_table_schema("stg_faostat_emissions", max_bytes=24000)
    assert "Element value samples:" in text
    assert "Item value samples:" in text
    assert "Unit value samples:" in text
    assert "Emissions (CO2eq) (AR5)" in text
    assert "Enteric Fermentation" in text
    assert "t CO2eq/ha" in text
    assert "one metric" in text.lower() or "exactly one" in text.lower() or "never SUM" in text
    assert "sector aggregates" in text.lower() or "primary" in text.lower()


def test_food_balances_schema_pack_surfaces_samples_and_guidance() -> None:
    text = format_table_schema("stg_faostat_food_balances", max_bytes=48000)
    assert "Product value samples:" in text
    assert "Element value samples:" in text
    assert "Unit value samples:" in text
    assert "Domestic supply quantity" in text
    assert "Potatoes" in text
    assert "kcal/cap/d" in text
    assert (
        "exactly one" in text.lower()
        or "one metric" in text.lower()
        or "never SUM" in text
    )
    assert "aggregate" in text.lower() or "Grand Total" in text
    assert "primary" in text.lower()


def test_investment_asti_schema_pack_surfaces_samples_and_guidance() -> None:
    text = format_table_schema("stg_faostat_investment_asti", max_bytes=48000)
    assert "Donor value samples:" in text
    assert "Purpose value samples:" in text
    assert "Item value samples:" in text
    assert "Element value samples:" in text
    assert "Indicator value samples:" in text
    assert "Unit value samples:" in text
    assert "Bill & Melinda Gates Foundation" in text
    assert "Agricultural research" in text
    assert "Value US$, 2015 prices" in text
    assert "million USD" in text
    assert (
        "exactly one" in text.lower()
        or "one metric" in text.lower()
        or "never SUM" in text
    )
    assert "All Donors" in text or "rollup" in text.lower()
    assert "ASTI" in text or "ODA" in text


def test_land_inputs_schema_pack_surfaces_samples_and_guidance() -> None:
    text = format_table_schema("stg_faostat_land_inputs", max_bytes=48000)
    assert "Item value samples:" in text
    assert "Element value samples:" in text
    assert "Unit value samples:" in text
    assert "Agricultural Use" in text
    assert "Nutrient nitrogen N (total)" in text
    assert "1000 ha" in text
    assert (
        "exactly one" in text.lower()
        or "one metric" in text.lower()
        or "never SUM" in text
    )
    assert "fertilizer" in text.lower() or "land" in text.lower()
    assert "primary" in text.lower() or "rollup" in text.lower()


def test_population_employment_schema_pack_surfaces_samples_and_guidance() -> None:
    text = format_table_schema("stg_faostat_population_employment", max_bytes=32000)
    assert "Element value samples:" in text
    assert "Indicator value samples:" in text
    assert "Source value samples:" in text
    assert "Total Population - Both sexes" in text
    assert "Share of employment in agriculture in total employment" in text
    assert "Labour force survey" in text
    assert (
        "exactly one" in text.lower()
        or "one metric" in text.lower()
        or "never SUM" in text
    )
    assert "population" in text.lower() and "employment" in text.lower()
    assert "15+" in text or "age" in text.lower()


def test_prices_schema_pack_surfaces_samples_and_guidance() -> None:
    text = format_table_schema("stg_faostat_prices", max_bytes=48000)
    assert "Product value samples:" in text
    assert "Element value samples:" in text
    assert "Currency value samples:" in text
    assert "Unit value samples:" in text
    assert "Producer Price (LCU/tonne)" in text
    assert "Wheat" in text
    assert "Naira" in text
    assert "LCU" in text
    assert (
        "exactly one" in text.lower()
        or "one metric" in text.lower()
        or "never SUM" in text
    )
    assert "primary" in text.lower() or "rollup" in text.lower()
    assert "index" in text.lower() or "PPI" in text or "producer" in text.lower()


def test_multi_table_pack_includes_both_schemas() -> None:
    hints, _ = pack_selected_table_hints(
        ["stg_faostat_production", "stg_yield_raw_data"],
        max_bytes=12000,
    )
    assert len(hints) >= 2
    joined = "\n".join(hints)
    assert "stg_faostat_production" in joined
    assert "stg_yield_raw_data" in joined
    assert "Columns:" in joined


def test_multi_table_pack_shows_join_when_both_selected() -> None:
    hints, _ = pack_selected_table_hints(
        ["stg_yield_raw_data", "stg_faostat_production"],
        max_bytes=12000,
    )
    joined = "\n".join(hints)
    assert "joins_with" in joined.lower()
    assert "stg_faostat_production" in joined


def test_reasoner_index_mentions_rels() -> None:
    text, _ = format_reasoner_index(max_bytes=20000)
    assert "rels=" in text
    assert "stg_yield_raw_data" in text


def test_load_table_schema_by_fqn() -> None:
    schema = load_table_schema("opentrace-prod-5ga4.staging_dev.stg_wfp_vampire_prices")
    assert schema is not None
    assert "description" in schema
    assert "semantic_relationships" in schema


def test_pack_selected_hints_respects_budget() -> None:
    hints, truncated = pack_selected_table_hints(
        ["stg_yield_raw_data", "stg_faostat_production", "stg_fews_food_security"],
        max_bytes=1500,
    )
    assert hints
    assert utf8_len("\n".join(hints)) <= 1500
    assert truncated in (True, False)


def test_byte_truncate_and_pack_lines() -> None:
    out, was = truncate_utf8("abcdefghij", 5)
    assert was
    assert utf8_len(out) <= 5
    text, trunc = pack_lines(["aaaa", "bbbb", "cccc"], max_bytes=10)
    assert trunc
    assert utf8_len(text) <= 10


def test_trim_bq_result_contents() -> None:
    rows = [{"content": "x" * 100}, {"content": "y" * 100}, {"content": "z" * 100}]
    out, truncated = trim_bq_result_contents(rows, max_bytes=150)
    assert truncated
    assert out
    assert utf8_len("".join(str(r.get("content") or "") for r in out)) <= 150


def test_validate_sql_staging_only() -> None:
    ok = _validate_sql(
        "SELECT country FROM `opentrace-prod-5ga4.staging_dev.stg_yield_raw_data`",
        {"staging_dev"},
        10,
    )
    assert ok is not None
    assert "LIMIT" in ok.upper()

    bad = _validate_sql(
        "SELECT * FROM `opentrace-prod-5ga4.raw_dev.yield_raw_data`",
        {"staging_dev"},
        10,
    )
    assert bad is None

    bronze = _validate_sql(
        "SELECT * FROM `opentrace-prod-5ga4.bronze.yield_raw_data`",
        {"staging_dev"},
        10,
    )
    assert bronze is None


def test_datasets_config_defaults_to_staging_dev(monkeypatch) -> None:
    monkeypatch.delenv("BQ_DATASET_SILVER", raising=False)
    cfg = _get_datasets_config()
    assert cfg.get("staging") == "staging_dev"


def test_reasoner_empty_llm_fails_closed() -> None:
    with patch("ml.rag.chatbot.bq_sql_reasoner.llm_chat_complete", return_value=""):
        plan = reason_bq_sql_plan("maize yield by district in Ghana", plan_type="Farmers")
    assert plan["skip_bq"] is True
    assert plan["selected_tables"] == []
    assert plan["rationale"] == "reasoner_unavailable"
    assert plan["table_hints"] == []


def test_reasoner_invalid_json_fails_closed() -> None:
    with patch("ml.rag.chatbot.bq_sql_reasoner.llm_chat_complete", return_value="not json"):
        plan = reason_bq_sql_plan("What are prices?")
    assert plan["skip_bq"] is True
    assert plan["selected_tables"] == []
    assert plan["rationale"] == "invalid_plan"


def test_reasoner_uses_mock_llm_json() -> None:
    payload = (
        '{"selected_tables":["stg_wfp_vampire_prices"],'
        '"query_intents":[{"goal":"prices","tables":["stg_wfp_vampire_prices"],'
        '"filters":"country=Ghana","notes":""}],'
        '"skip_bq":false,"rationale":"prices question"}'
    )
    with patch("ml.rag.chatbot.bq_sql_reasoner.llm_chat_complete", return_value=payload):
        plan = reason_bq_sql_plan("What are maize prices in Ghana?", plan_type="Farmers")
    assert plan["selected_tables"] == ["stg_wfp_vampire_prices"]
    assert plan["table_hints"]
    assert plan["skip_bq"] is False


def test_reasoner_rejects_unknown_only_tables() -> None:
    payload = (
        '{"selected_tables":["not_a_real_table"],'
        '"query_intents":[],"skip_bq":false,"rationale":"x"}'
    )
    with patch("ml.rag.chatbot.bq_sql_reasoner.llm_chat_complete", return_value=payload):
        plan = reason_bq_sql_plan("HDI for Kenya?")
    assert plan["selected_tables"] == []
    assert plan["skip_bq"] is True
    assert plan["rationale"] == "invalid_plan"


def test_reasoner_keeps_valid_among_unknown() -> None:
    payload = (
        '{"selected_tables":["not_a_real_table","stg_africa_hdi"],'
        '"query_intents":[],"skip_bq":false,"rationale":"x"}'
    )
    with patch("ml.rag.chatbot.bq_sql_reasoner.llm_chat_complete", return_value=payload):
        plan = reason_bq_sql_plan("HDI for Kenya?")
    assert plan["selected_tables"] == ["stg_africa_hdi"]
    assert plan["skip_bq"] is False


def test_no_fallback_sql_method() -> None:
    assert not hasattr(BQRetriever, "_fallback_sql")
    assert not hasattr(bq_retriever, "_fallback_sql")
