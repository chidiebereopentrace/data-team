from ml.rag.chatbot.mart_coverage import Coverage, CoverageEntry, load_coverage, route_tables


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


def test_load_coverage_is_cached() -> None:
    first = load_coverage
    assert first.cache_info().maxsize == 1
