"""Unit tests for FAOSTAT trend companion direction alignment."""
from __future__ import annotations

from ml.rag.chatbot.bq_trend_companion import (
    align_trend_directions,
    build_faostat_trend_companion_sql,
    parse_trend_companion_rows,
)


def test_build_faostat_trend_companion_sql() -> None:
    sql = build_faostat_trend_companion_sql(
        project_id="proj",
        dataset="staging_dev",
        country_name="Nigeria",
        focal_year=2020,
    )
    assert "stg_faostat_production" in sql
    assert "Nigeria" in sql
    assert "2019" in sql
    assert "2021" in sql
    assert "'Production'" in sql
    assert "'Yield'" in sql


def test_production_and_yield_both_increasing() -> None:
    rows = [
        {"element": "Production", "year": 2019, "total": 100.0},
        {"element": "Production", "year": 2021, "total": 120.0},
        {"element": "Yield", "year": 2019, "total": 2000.0},
        {"element": "Yield", "year": 2021, "total": 2200.0},
    ]
    parsed = parse_trend_companion_rows(rows, focal_year=2020)
    aligned = align_trend_directions(parsed)
    assert aligned["direction"] == "increasing"
    assert aligned["trend_mixed"] is False
    assert aligned["prior_value"] == 100.0
    assert aligned["value"] == 120.0


def test_mixed_production_yield_signals() -> None:
    rows = [
        {"element": "Production", "year": 2019, "total": 100.0},
        {"element": "Production", "year": 2021, "total": 120.0},
        {"element": "Yield", "year": 2019, "total": 2200.0},
        {"element": "Yield", "year": 2021, "total": 2000.0},
    ]
    parsed = parse_trend_companion_rows(rows, focal_year=2020)
    aligned = align_trend_directions(parsed)
    assert aligned["direction"] == "unknown"
    assert aligned["trend_mixed"] is True
