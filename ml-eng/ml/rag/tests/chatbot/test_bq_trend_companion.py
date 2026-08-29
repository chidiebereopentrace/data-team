"""Unit tests for mart production trend companion direction alignment."""
from __future__ import annotations

from ml.rag.chatbot.bq_trend_companion import (
    align_trend_directions,
    build_mart_production_trend_companion_sql,
    maybe_attach_ranking_trend,
    parse_trend_companion_rows,
)


def test_build_mart_production_trend_companion_sql() -> None:
    sql = build_mart_production_trend_companion_sql(
        project_id="proj",
        dataset="mart_dev",
        country_iso3="NGA",
        focal_year=2020,
    )
    assert "fct_production" in sql
    assert "mart_dev" in sql
    assert "NGA" in sql
    assert "2019" in sql
    assert "2021" in sql
    assert "production_grain = 'physical'" in sql
    assert "'Production'" in sql
    assert "'Yield'" in sql
    assert "country_iso3" in sql


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


def test_maybe_attach_ranking_trend_uses_country_iso3(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_fetch(**kwargs):
        captured["country_iso3"] = kwargs["country_iso3"]
        return {
            "direction": "increasing",
            "value": 120.0,
            "prior_value": 100.0,
            "magnitude": 20.0,
            "sql": "SELECT 1",
        }

    monkeypatch.setattr(
        "ml.rag.chatbot.bq_trend_companion.fetch_mart_production_trend_companion",
        _fake_fetch,
    )
    meta = {
        "bq_enrichment": "ranked_table",
        "year": 2020,
        "ranked_rows": [
            {
                "label": "NGA",
                "value": 5000,
                "raw_row": {"country_iso3": "NGA", "total": 5000},
            }
        ],
    }
    out = maybe_attach_ranking_trend(
        meta,
        sql="SELECT country_iso3, SUM(value) AS total FROM `p.mart_dev.fct_production`",
        template="faostat_production_rank",
    )
    assert captured["country_iso3"] == "NGA"
    assert out["direction"] == "increasing"
    assert out["magnitude"] == 20.0
