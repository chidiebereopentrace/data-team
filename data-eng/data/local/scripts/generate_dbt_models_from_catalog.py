#!/usr/bin/env python3
"""
Generate dbt models from the BigQuery schema catalog and the staging taxonomy.

Flat layers (raw_dev, mart_dev, optional landing):
  dbt/models/<layer>/<table_name>.sql
  select * from {{ source('<layer>', '<table_name>') }}

staging_dev (domain-grouped tree):
  Driven by staging_dev_taxonomy.yml — domain stg_* paths under
  dbt/models/staging_dev/. Every mapped model uses source('raw_dev', …)
  only (no landing / cross-layer jumps).

  Taxonomy entries use `tables:` (list). One table => passthrough stub;
  multiple => CTE + union all stub with `_raw_table`. Null/empty => unmapped
  placeholder. Legacy single `table:` is still accepted as a one-element list.

Existing model files are never overwritten.

Usage (from data-eng / repo root, after catalog + sources):

  python data/local/scripts/bq_schema_catalog.py
  python data/local/scripts/generate_dbt_sources.py
  python data/local/scripts/generate_dbt_models_from_catalog.py
  python data/local/scripts/generate_dbt_models_from_catalog.py --staging-only
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "docs" / "bq_schema_catalog.json"
DBT_MODELS_ROOT = REPO_ROOT / "dbt" / "models"
TAXONOMY_PATH = Path(__file__).resolve().parent / "staging_dev_taxonomy.yml"

FLAT_SOURCES_DEFAULT = ("raw_dev", "mart_dev")
STAGING_SOURCE = "raw_dev"

MAPPED_STUB_TEMPLATE = """{{{{ config(materialized='{materialized}', enabled={enabled}) }}}}

select
    *
from {{{{ source('{source_name}', '{table_name}') }}}}
"""

UNMAPPED_STUB_TEMPLATE = """{{{{ config(materialized='{materialized}', enabled=false) }}}}

-- Unmapped: no raw_dev tables in staging_dev_taxonomy.yml yet.
-- Set `tables:` in the taxonomy and re-run the generator (this file is not overwritten).
select
    cast(null as string) as _unmapped
"""


def safe_model_filename(table_name: str) -> str:
    """Return a safe filename for the given table name."""
    return f"{table_name}.sql"


def cte_alias(table_name: str, index: int) -> str:
    """Build a SQL-safe CTE name from a BigQuery table id."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", table_name).strip("_").lower()
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"src_{cleaned}" if cleaned else f"src_{index}"
    # Keep aliases unique and bounded for readability.
    return f"t{index}_{cleaned[:48]}"


def normalize_tables(entry: dict) -> list[str] | None:
    """
    Resolve taxonomy tables for an entry.
    Returns None for unmapped; otherwise a non-empty list of table ids.
    Accepts `tables:` (preferred) or legacy `table:`.
    """
    if "tables" in entry:
        raw = entry.get("tables")
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            raise SystemExit(
                f"staging_dev_taxonomy.yml: `tables` must be a list or null "
                f"(path={entry.get('path')})"
            )
        tables = [str(t).strip() for t in raw if t is not None and str(t).strip()]
        return tables or None

    legacy = entry.get("table")
    if legacy is None or (isinstance(legacy, str) and not legacy.strip()):
        return None
    return [str(legacy).strip()]


def render_multi_table_stub(materialized: str, table_names: list[str]) -> str:
    """CTE per raw_dev table, union all with _raw_table discriminator."""
    lines = [
        f"{{{{ config(materialized='{materialized}', enabled=false) }}}}",
        "",
        "-- Domain stub: refine column lists before enabling (schemas may differ).",
        "with",
    ]
    cte_names: list[str] = []
    for i, table_name in enumerate(table_names):
        alias = cte_alias(table_name, i)
        cte_names.append(alias)
        comma = "," if i < len(table_names) - 1 else ""
        lines.append(f"{alias} as (")
        lines.append("    select")
        lines.append("        *,")
        lines.append(f"        '{table_name}' as _raw_table")
        lines.append(f"    from {{{{ source('{STAGING_SOURCE}', '{table_name}') }}}}")
        lines.append(f"){comma}")
    lines.append("")
    union_parts = [f"select * from {name}" for name in cte_names]
    lines.append("\nunion all\n".join(union_parts))
    lines.append("")
    return "\n".join(lines)


def load_catalog() -> dict:
    if not CATALOG_PATH.exists():
        raise SystemExit(
            f"Catalog not found at {CATALOG_PATH.relative_to(REPO_ROOT)}. "
            "Run data/local/scripts/bq_schema_catalog.py first."
        )
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def load_taxonomy() -> list[dict]:
    if not TAXONOMY_PATH.exists():
        raise SystemExit(
            f"Taxonomy not found at {TAXONOMY_PATH.relative_to(REPO_ROOT)}. "
            "Expected staging_dev_taxonomy.yml next to this script."
        )
    data = yaml.safe_load(TAXONOMY_PATH.read_text(encoding="utf-8")) or {}
    models = data.get("models") or []
    if not isinstance(models, list):
        raise SystemExit("staging_dev_taxonomy.yml: `models` must be a list")
    return models


def generate_flat_models(
    catalog: dict,
    sources: tuple[str, ...] | list[str],
    materialized: str,
) -> tuple[list[str], list[str]]:
    """Generate flat passthrough stubs for the given source layers."""
    datasets: dict = catalog.get("datasets", {})
    created: list[str] = []
    skipped_existing: list[str] = []

    for source_name in sources:
        source_info = datasets.get(source_name)
        if not source_info:
            continue
        tables = source_info.get("tables", []) or []
        if not tables:
            continue

        source_dir = DBT_MODELS_ROOT / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        for t in tables:
            table_name = t.get("table")
            if not table_name:
                continue
            model_path = source_dir / safe_model_filename(table_name)
            rel = model_path.relative_to(REPO_ROOT)
            if model_path.exists():
                skipped_existing.append(str(rel))
                continue
            content = MAPPED_STUB_TEMPLATE.format(
                materialized=materialized,
                enabled="false",
                source_name=source_name,
                table_name=table_name,
            )
            model_path.write_text(content, encoding="utf-8")
            created.append(str(rel))

    return created, skipped_existing


def generate_staging_models(materialized: str) -> tuple[list[str], list[str], list[str]]:
    """
    Generate domain-grouped staging_dev stubs from taxonomy.
    Returns (created, skipped_existing, unmapped_paths).
    """
    created: list[str] = []
    skipped_existing: list[str] = []
    unmapped: list[str] = []

    staging_root = DBT_MODELS_ROOT / "staging_dev"
    staging_root.mkdir(parents=True, exist_ok=True)

    for entry in load_taxonomy():
        rel_path = (entry.get("path") or "").strip().replace("\\", "/")
        if not rel_path:
            continue
        model_path = staging_root / rel_path
        rel = model_path.relative_to(REPO_ROOT)

        if model_path.exists():
            skipped_existing.append(str(rel))
            continue

        model_path.parent.mkdir(parents=True, exist_ok=True)
        tables = normalize_tables(entry)

        if not tables:
            content = UNMAPPED_STUB_TEMPLATE.format(materialized=materialized)
            unmapped.append(str(rel))
        elif len(tables) == 1:
            content = MAPPED_STUB_TEMPLATE.format(
                materialized=materialized,
                enabled="false",
                source_name=STAGING_SOURCE,
                table_name=tables[0],
            )
        else:
            content = render_multi_table_stub(materialized, tables)

        model_path.write_text(content, encoding="utf-8")
        created.append(str(rel))

    return created, skipped_existing, unmapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dbt models: flat raw_dev/mart_dev from catalog; "
            "domain-grouped staging_dev from staging_dev_taxonomy.yml "
            "(source raw_dev only)."
        )
    )
    parser.add_argument(
        "--materialized",
        choices=["view", "table", "incremental"],
        default="view",
        help="Materialization for generated models (default: view)",
    )
    parser.add_argument(
        "--include-landing",
        action="store_true",
        help="Also generate disabled stubs under dbt/models/landing.",
    )
    parser.add_argument(
        "--staging-only",
        action="store_true",
        help="Only generate staging_dev models from the taxonomy.",
    )
    parser.add_argument(
        "--skip-staging",
        action="store_true",
        help="Skip staging_dev taxonomy generation (flat layers only).",
    )
    args = parser.parse_args()

    if args.staging_only and args.skip_staging:
        raise SystemExit("Cannot combine --staging-only and --skip-staging")

    flat_created: list[str] = []
    flat_skipped: list[str] = []
    stg_created: list[str] = []
    stg_skipped: list[str] = []
    stg_unmapped: list[str] = []

    if not args.staging_only:
        catalog = load_catalog()
        sources = list(FLAT_SOURCES_DEFAULT)
        if args.include_landing:
            sources = ["landing", *sources]
        flat_created, flat_skipped = generate_flat_models(
            catalog, sources, args.materialized
        )

    if not args.skip_staging:
        stg_created, stg_skipped, stg_unmapped = generate_staging_models(
            args.materialized
        )

    print("=== generate_dbt_models_from_catalog ===")
    if not args.staging_only:
        print(f"Flat layers: created={len(flat_created)} skipped={len(flat_skipped)}")
        for path in flat_created:
            print(f"  + {path}")
    if not args.skip_staging:
        print(
            f"staging_dev: created={len(stg_created)} skipped={len(stg_skipped)} "
            f"unmapped={len(stg_unmapped)}"
        )
        for path in stg_created:
            print(f"  + {path}")
        if stg_unmapped:
            print("Unmapped staging stubs (tables: null):")
            for path in stg_unmapped:
                print(f"  ? {path}")
    if (
        not flat_created
        and not stg_created
        and (flat_skipped or stg_skipped)
    ):
        print("No new models created (all already existed).")


if __name__ == "__main__":
    main()
