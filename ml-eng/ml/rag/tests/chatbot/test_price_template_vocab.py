"""Price template SQL uses only mart vocabulary: product keys, sources, types, metrics."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from ml.rag.chatbot.capability_registry import resolve_capability
from ml.rag.chatbot.price_sql import plan_price_template
from ml.rag.chatbot.price_vocab import names_price_product, price_sql_vocab_errors
from ml.rag.chatbot.sql_ast_validate import parse_select
from ml.rag.chatbot.turn_contract import TimeSpec, TurnContract
from ml.rag.retrievers.bq_retriever import BQRetriever


def _plan(query: str, countries: list[str], year: str | None = None) -> dict:
    hit = plan_price_template(
        query=query,
        blob=query,
        country_labels=countries,
        time_start=f"{year}-01-01" if year else None,
        time_end=f"{year}-12-31" if year else None,
        project_id="proj",
        dataset="mart_dev",
    )
    assert hit is not None
    return hit


@pytest.mark.parametrize(
    ("query", "country", "year"),
    [
        ("What was the average retail maize price in Nigeria in 2024?", "Nigeria", "2024"),
        ("Mean wholesale rice price for Kenya in 2023", "Kenya", "2023"),
        ("Average sorghum price in Ethiopia during 2022", "Ethiopia", "2022"),
        ("Average millet price in Niger in 2023", "Niger", "2023"),
        ("Average cassava price in Ghanaian markets in 2023", "Ghana", "2023"),
        ("Average tomato price in Nigerian markets in 2023", "Nigeria", "2023"),
        ("What is the current price of maize in Kenya?", "Kenya", None),
    ],
)
def test_price_sql_is_keyed_and_labelled_from_vocab(query: str, country: str, year: str | None) -> None:
    hit = _plan(query, [country], year)
    sql = hit["sql"]
    assert hit["template"] == "mart_price"
    assert "product_key IN (" in sql
    assert "price_source =" in sql or "price_source IN (" in sql
    assert "b.unit, b.currency" in sql
    assert "d.product_name" in sql
    assert price_sql_vocab_errors(sql) == []
    if year:
        assert f"year = {year}" in sql
    else:
        assert "QUALIFY" in sql


def test_wholesale_question_filters_wholesale() -> None:
    sql = _plan("Mean wholesale rice price for Kenya in 2023", ["Kenya"], "2023")["sql"]
    assert "price_type = 'Wholesale'" in sql


def test_rwanda_rice_can_plan_the_requested_year() -> None:
    hit = _plan("What was the price of rice in Rwanda in 2025?", ["Rwanda"], "2025")
    assert hit["template"] == "mart_price"
    assert "country_iso3 IN ('RWA')" in hit["sql"]
    assert "year = 2025" in hit["sql"]


def test_multi_country_multi_year_groups_by_year() -> None:
    hit = plan_price_template(
        query="Compare wholesale rice prices in Kenya and Uganda since 2020",
        blob="Compare wholesale rice prices in Kenya and Uganda since 2020",
        country_labels=["Kenya", "Uganda"],
        time_start="2020-01-01",
        time_end="2026-12-31",
        project_id="proj",
        dataset="mart_dev",
    )
    assert hit is not None
    sql = hit["sql"]
    assert "country_iso3 IN ('KEN', 'UGA')" in sql
    assert "year BETWEEN 2020 AND 2026" in sql
    assert "b.year" in sql


@pytest.mark.parametrize("wording", ["the price of food", "food prices"])
def test_policy_question_food_price_series_use_staples(wording: str) -> None:
    query = f"How did agricultural policies affect farmer livelihoods and {wording} since 2015?"
    hit = _plan(query, ["Ghana", "Liberia", "Mali"])

    assert hit["template"] == "mart_price"
    assert hit["product_names"]
    assert "wage" not in hit["sql"].lower()
    assert "country_iso3 IN (" in hit["sql"]


def test_regional_food_basket_sql_parses_with_apostrophes_in_mart_values(monkeypatch) -> None:
    query = "How did agricultural policies affect farmer livelihoods and food prices in West Africa since 2015?"
    hit = _plan(query, ["Mauritania", "Côte d'Ivoire", "Ghana"], "2024")
    assert hit["template"] == "mart_price"
    _, error = parse_select(hit["sql"])
    assert error is None
    monkeypatch.setenv("RAG_BQ_SQL_DRY_RUN", "on")
    client = Mock()
    client.query.return_value.total_bytes_processed = 0
    validated, error = BQRetriever(project_id="proj", nl2sql_enabled=False)._prepare_sql(
        hit["sql"],
        question=query,
        table_hints=[],
        selected_tables={"fct_prices"},
        allowed_datasets={"mart_dev"},
        limit=60,
        client=client,
        crop_required=False,
        sql_source="template",
    )
    assert error is None
    assert validated is not None


def test_source_follows_requested_years() -> None:
    hit = _plan("What was the mean onion price in Senegal in 2022?", ["Senegal"], "2022")
    assert hit["price_source"] == "faostat"
    assert hit["widened"] == ""


def test_year_outside_coverage_widens_to_latest_with_note() -> None:
    hit = _plan("Average millet price in Niger in 1950", ["Niger"], "1950")
    assert "QUALIFY" in hit["sql"]
    assert "1950" in hit["widened"]


def test_product_not_held_fails_closed_with_closest() -> None:
    hit = _plan("Average maize flour price in Ethiopia in 2023", ["Ethiopia"], "2023")
    assert hit["template"] == "price_gap"
    assert hit["sql"] == ""
    assert "Ethiopia" in hit["gap_message"]


def test_named_region_splits_rows_into_place_and_rest_of_country() -> None:
    hit = _plan("What is the current retail price of maize in markets around Bamako, Mali?", ["Mali"])
    sql = hit["sql"]
    assert hit["places"] == ["admin1:Bamako"]
    assert "CASE WHEN market_key IN (" in sql
    assert "THEN 'Bamako' ELSE 'Rest of Mali' END AS admin_1" in sql
    assert "b.admin_1" in sql
    assert price_sql_vocab_errors(sql) == []


def test_region_without_the_product_says_so() -> None:
    hit = _plan("Are tomato prices going up or down at markets near Kano, Nigeria?", ["Nigeria"])
    assert hit["places"] == ["admin1:Kano"]
    assert "No Kano price rows" in hit["widened"]


def test_country_only_question_has_no_place_split() -> None:
    hit = _plan("What is the current price of maize in Kenya?", ["Kenya"])
    assert hit["places"] == []
    assert "CASE WHEN market_key" not in hit["sql"]


def test_multi_country_question_has_no_place_split() -> None:
    hit = plan_price_template(
        query="Maize prices in Nairobi and Kampala in 2023",
        blob="Maize prices in Nairobi and Kampala in 2023",
        country_labels=["Kenya", "Uganda"],
        time_start="2023-01-01",
        time_end="2023-12-31",
        project_id="proj",
        dataset="mart_dev",
    )
    assert hit is not None
    assert hit["places"] == []


def test_named_price_type_is_pinned() -> None:
    hit = _plan("What was the average retail maize price in Nigeria in 2024?", ["Nigeria"], "2024")
    assert hit["price_type"] == "Retail"
    assert "price_type = 'Retail'" in hit["sql"]


@pytest.mark.parametrize("question,country,year,price_predicate", [
    ("whats the price of maize in kenya this year", "Kenya", "2026", "price_type IN ('Retail', 'Wholesale')"),
    ("Average retail maize price in Nigeria in 2024", "Nigeria", "2024", "price_type = 'Retail'"),
    ("Mean wholesale rice price for Kenya in 2023", "Kenya", "2023", "price_type = 'Wholesale'"),
])
def test_price_template_passes_warehouse_validation_in_one_call(
    monkeypatch: pytest.MonkeyPatch, question: str, country: str, year: str, price_predicate: str,
) -> None:
    monkeypatch.setenv("RAG_BQ_SQL_DRY_RUN", "on")
    sql = _plan(question, [country], year)["sql"]
    client = Mock()
    client.query.return_value.total_bytes_processed = 0
    validated, error = BQRetriever(project_id="proj", nl2sql_enabled=False)._prepare_sql(
        sql,
        question=question,
        table_hints=[],
        selected_tables={"fct_prices"},
        allowed_datasets={"mart_dev"},
        limit=60,
        client=client,
        sql_source="template",
        time_start=f"{year}-01-01",
        time_end=f"{year}-12-31",
        primary_measures=["market_price"],
        geography=[country],
    )

    assert error is None, error
    assert validated is not None
    assert price_predicate in validated
    assert "GROUP BY b.country_iso3, d.product_name, b.price_type, b.price_source, b.unit, b.currency" in validated
    client.query.assert_called_once()


def test_unnamed_price_source_keeps_every_held_source() -> None:
    hit = _plan("Average maize price in Nigeria in 2023", ["Nigeria"], "2023")
    sql = hit["sql"]
    sources = hit["price_source"]
    if isinstance(sources, list) and len(sources) > 1:
        assert "price_source IN (" in sql
        for src in sources:
            assert f"'{src}'" in sql
    else:
        assert "price_source =" in sql


def test_multi_currency_comparison_carries_no_ranking_note() -> None:
    hit = plan_price_template(
        query="Compare cassava prices in Ghana vs Nigeria for 2023",
        blob="Compare cassava prices in Ghana vs Nigeria for 2023",
        country_labels=["Ghana", "Nigeria"],
        time_start="2023-01-01",
        time_end="2023-12-31",
        project_id="proj",
        dataset="mart_dev",
    )
    assert hit is not None
    assert "GHS" in hit["widened"] and "NGN" in hit["widened"]
    assert "exchange rate" in hit["widened"]


def test_volatility_question_is_a_price_template() -> None:
    hit = _plan("Was maize more volatile than rice in Nigerian markets during 2023?", ["Nigeria"], "2023")
    assert hit["template"] == "mart_price"


def test_warehouse_products_count_as_named_commodity() -> None:
    assert names_price_product("Average palm oil price in Nigerian markets in 2023?")
    assert not names_price_product("How is the weather in Nigeria?")


@pytest.mark.parametrize("geo_grain", ["point", "admin1", "country", "region", "africa"])
@pytest.mark.parametrize("time_grain", ["year", "year_range", "latest"])
def test_market_price_is_served_at_every_geo_and_time_grain(geo_grain: str, time_grain: str) -> None:
    contract = resolve_capability(
        TurnContract(measure_id="market_price", geo_grain=geo_grain, time_spec=TimeSpec(grain=time_grain))
    )
    assert contract.serve_status == "served"
    assert contract.sql_plan["template"] == "mart_price"


def test_non_price_question_is_not_a_price_template() -> None:
    assert (
        plan_price_template(
            query="Maize production in Nigeria in 2022",
            blob="Maize production in Nigeria in 2022",
            country_labels=["Nigeria"],
            time_start="2022-01-01",
            time_end="2022-12-31",
            project_id="proj",
            dataset="mart_dev",
        )
        is None
    )
