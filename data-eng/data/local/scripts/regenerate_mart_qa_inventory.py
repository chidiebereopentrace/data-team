#!/usr/bin/env python3
"""Regenerate data-eng/docs/_mart_qa_inventory_raw.md from mart_dev fact tables."""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "data" / "local" / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "data" / "local" / "scripts"))


def _load_dotenv() -> None:
    env_file = REPO_ROOT / "data" / "local" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and not (os.environ.get(key) or "").strip():
                os.environ[key] = value


FACTS = [
    "fct_prices",
    "fct_food_security",
    "fct_production",
    "fct_yield",
    "fct_climate",
    "fct_air_quality",
    "fct_soil_health",
    "fct_investment",
    "fct_gender_inclusion",
    "fct_animal_health",
    "fct_insurance",
    "fct_biodiversity",
    "fct_vegetation",
    "fct_protected_areas",
    "fct_germplasm",
    "fct_food_balance",
    "fct_trade",
]


def main() -> int:
    _load_dotenv()
    from google.cloud import bigquery

    project = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4")
    mart = os.environ.get("BQ_DATASET_GOLD", "mart_dev")
    client = bigquery.Client(project=project)
    out_path = REPO_ROOT / "docs" / "_mart_qa_inventory_raw.md"

    lines = [
        f"> **Snapshot note:** Regenerated from `{project}.{mart}` ({date.today().isoformat()}) "
        "after geo follow-ups (geoBoundaries ADM0, germplasm ArcGIS backfill, gender Chagos doc). "
        "See [MART_QA_NOTES.md](./MART_QA_NOTES.md).",
        "",
    ]

    for fact in FACTS:
        sql = f"""
        SELECT
          source_natural_key,
          COUNT(*) AS n,
          COUNTIF(source_key IS NULL) AS src_null,
          COUNTIF(geography_key IS NULL) AS geo_null,
          COUNTIF(as_of_date IS NULL) AS asof_null
        FROM `{project}.{mart}.{fact}`
        GROUP BY 1
        ORDER BY n DESC
        """
        rows = list(client.query(sql).result())
        total = sum(r.n for r in rows)
        lines.append(f"## {fact} (total={total:,})")
        lines.append("| source_natural_key | n | src_null | geo_null | asof_null | geo_null_pct |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in rows:
            pct = 0.0 if row.n == 0 else 100.0 * row.geo_null / row.n
            key = row.source_natural_key or ""
            lines.append(
                f"| `{key}` | {row.n:,} | {row.src_null:,} | {row.geo_null:,} | {row.asof_null:,} | {pct:.1f}% |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
