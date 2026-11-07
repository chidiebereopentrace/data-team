"""Price template SQL uses only mart vocabulary: product keys, sources, types, metrics."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.capability_registry import resolve_capability
from ml.rag.chatbot.price_sql import plan_price_template
from ml.rag.chatbot.price_vocab import names_price_product, price_sql_vocab_errors
from ml.rag.chatbot.turn_contract import TimeSpec, TurnContract


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
    assert "price_source = " in sql
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


@pytest.mark.parametrize(
    ("query", "country", "year"),
    [
        ("Average maize price in Tanzania in 2023", "Tanzania", "2023"),
        ("What was the mean onion price in Senegal in 2022?", "Senegal", "2022"),
    ],
)
def test_price_type_is_always_pinned(query: str, country: str, year: str) -> None:
    hit = _plan(query, [country], year)
    assert hit["price_type"]
    assert f"price_type = '{hit['price_type']}'" in hit["sql"]


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
