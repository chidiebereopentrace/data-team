"""Contract tests for categorical-filter-first mart YAML profiling."""
from __future__ import annotations

from pathlib import Path

import yaml

YAML_DIR = Path(__file__).resolve().parents[2] / "bq_mart_tables_yaml_files"

MEASURE_NUMERIC_COLS = frozenset(
    {
        "value",
        "production_qty",
        "yield_value",
        "area_harvested",
        "total_production_qty",
        "total_population_affected",
        "avg_population_affected",
        "avg_pct_phase3",
        "avg_pct_phase4",
        "avg_pct_phase5",
        "record_count",
        "sum_value",
        "total",
    }
)


def _load_yaml(name: str) -> dict:
    path = YAML_DIR / name
    assert path.is_file(), f"missing {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_food_security_yaml_has_no_measure_value_samples() -> None:
    doc = _load_yaml("agg_food_security_monthly.yml")
    for col in MEASURE_NUMERIC_COLS:
        key = f"{col}_value_samples"
        assert key not in doc, f"{key} must not appear on agg_food_security_monthly"


def test_food_security_yaml_has_categorical_filter_samples() -> None:
    doc = _load_yaml("agg_food_security_monthly.yml")
    assert doc.get("country_name_value_samples"), "country_name_value_samples required"
    assert doc.get("phase_name_value_samples"), "phase_name_value_samples required"
    assert doc.get("scenario_name_value_samples"), "scenario_name_value_samples required"


def test_agg_production_annual_has_country_and_product_samples() -> None:
    doc = _load_yaml("agg_production_annual.yml")
    assert doc.get("country_name_value_samples")
    assert doc.get("product_name_value_samples")
    assert "total_production_qty_value_samples" not in doc
