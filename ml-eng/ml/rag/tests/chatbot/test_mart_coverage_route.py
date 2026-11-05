from ml.rag.chatbot.mart_coverage import Coverage, CoverageEntry, clamp_year, load_coverage, route_tables


def test_route_tables_keeps_order_of_tables_that_hold_the_country(monkeypatch) -> None:
    cov = Coverage(
        entries=[
            CoverageEntry("fct_prices", "country_iso3", "KEN", "Kenya", "KEN", 2010, 2024),
            CoverageEntry("agg_prices_country_month", "country_iso3", "KEN", "Kenya", "KEN", 2010, 2024),
        ]
    )
    monkeypatch.setattr("ml.rag.chatbot.mart_coverage.load_coverage", lambda: cov)
    assert route_tables(["fct_production", "agg_prices_country_month", "fct_prices"], "KEN") == [
        "agg_prices_country_month",
        "fct_prices",
    ]


def test_clamp_year_caps_to_the_table_span(monkeypatch) -> None:
    cov = Coverage(
        entries=[
            CoverageEntry("fct_production", "country_iso3", "KEN", "Kenya", "KEN", 1961, 2024),
        ]
    )
    monkeypatch.setattr("ml.rag.chatbot.mart_coverage.load_coverage", lambda: cov)
    assert clamp_year("fct_production", "KEN", 2026) == 2024
    assert clamp_year("fct_production", "KEN", 2020) == 2020


def test_load_coverage_is_cached() -> None:
    first = load_coverage
    assert first.cache_info().maxsize == 1
