#!/usr/bin/env python3
"""Post-process mart YAML files: categorical-filter-first value_samples contract.

Run from repo root (no BigQuery required):
  python ml-eng/ml/rag/helpers/patch_mart_yaml_filter_contract.py

Removes measure/hash numeric value_samples; backfills human labels from dim YAMLs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
YAML_DIR = REPO_ROOT / "ml-eng" / "ml" / "rag" / "bq_mart_tables_yaml_files"

MEASURE_DISCRIMINATOR_COLS = frozenset(
    {
        "element",
        "metric",
        "production_grain",
        "price_type",
        "measure_type",
        "indicator",
        "trade_grain",
        "price_source",
        "phase_name",
        "scenario_name",
        "scenario_code",
        "phase_code",
    }
)

TIME_DIM_COLS = frozenset({"year", "time_key", "harvest_year", "observation_year", "mp_year", "month"})

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

SURROGATE_KEY_COLS = frozenset(
    c for c in MEASURE_DISCRIMINATOR_COLS
    if c.endswith("_key")
) | frozenset(
    {
        "geo_key",
        "geography_key",
        "product_key",
        "source_key",
        "classification_key",
        "scenario_key",
        "production_key",
        "yield_key",
        "price_key",
        "trade_key",
        "food_security_key",
    }
)

DIM_LABEL_BACKFILL: dict[tuple[str, str], tuple[str, str]] = {
    ("agg_food_security_monthly", "phase_name"): ("dim_classification", "phase_name"),
    ("agg_food_security_monthly", "scenario_name"): ("dim_scenario", "scenario_name"),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _column_names(payload: dict[str, Any]) -> set[str]:
    cols = payload.get("columns") or []
    return {str(c.get("name") or "").strip() for c in cols if str(c.get("name") or "").strip()}


def _should_keep_samples(col: str) -> bool:
    col_l = col.lower()
    if col_l in MEASURE_NUMERIC_COLS:
        return False
    if col_l in SURROGATE_KEY_COLS or col_l.endswith("_key"):
        return False
    if col_l in MEASURE_DISCRIMINATOR_COLS:
        return True
    if col_l in TIME_DIM_COLS:
        return True
    if col_l.endswith("_name") or col_l.endswith("_code") or col_l.endswith("_type"):
        return True
    if col_l.endswith("_grain") or col_l in ("unit", "country_iso3", "country_iso2"):
        return True
    return False


def _backfill_from_dim(table_id: str, col: str) -> list[str] | None:
    spec = DIM_LABEL_BACKFILL.get((table_id, col))
    if not spec:
        return None
    dim_table, dim_col = spec
    dim_path = YAML_DIR / f"{dim_table}.yml"
    if not dim_path.is_file():
        return None
    dim = _load_yaml(dim_path)
    samples = dim.get(f"{dim_col}_value_samples")
    if isinstance(samples, list) and samples:
        return [str(s) for s in samples if str(s).strip()]
    return None


def patch_table_yaml(table_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    col_names = _column_names(payload)
    out = dict(payload)
    keys_to_drop: list[str] = []

    for key in list(out.keys()):
        if not key.endswith("_value_samples"):
            continue
        col = key[: -len("_value_samples")]
        if col not in col_names:
            keys_to_drop.append(key)
            continue
        if not _should_keep_samples(col):
            keys_to_drop.append(key)
            if f"{col}_value_stats" not in out:
                out[f"{col}_value_stats"] = {
                    "distinct_count": len(out.get(key) or []),
                    "null_count": 0,
                    "is_truncated": False,
                    "patched": True,
                }

    for key in keys_to_drop:
        out.pop(key, None)

    for col in col_names:
        sample_key = f"{col}_value_samples"
        if sample_key in out:
            continue
        if not _should_keep_samples(col):
            continue
        backfill = _backfill_from_dim(table_id, col)
        if backfill:
            out[sample_key] = backfill[:50]

    return out


def main() -> int:
    if not YAML_DIR.is_dir():
        print(f"Missing {YAML_DIR}")
        return 1
    patched = 0
    for path in sorted(YAML_DIR.glob("*.yml")):
        table_id = path.stem
        payload = _load_yaml(path)
        if not payload:
            continue
        new_payload = patch_table_yaml(table_id, payload)
        if new_payload != payload:
            path.write_text(
                yaml.safe_dump(new_payload, sort_keys=False, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )
            patched += 1
    print(f"Patched {patched} YAML files in {YAML_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
