#!/usr/bin/env python3
"""
Dump BigQuery schema for configured datasets (default: landing, raw_dev, staging_dev, mart_dev; or BQ_DATASETS_INCLUDE) into a
single catalog. Use this to curate an aligned schema across BQ, dbt, and local DDL.

Outputs:
  - docs/bq_schema_catalog.json  (machine-readable: project, datasets, tables, columns)
  - docs/BIGQUERY_SCHEMA.md      (human-readable summary)

Usage:
  Set BQ_PROJECT and (optionally) BQ_DATASET_LANDING, BQ_DATASET_BRONZE, etc. in
  data/local/.env. From repo root:
    python data/local/scripts/bq_schema_catalog.py
  Options:
    --json-only    Write only the JSON catalog.
    --md-only      Write only the Markdown doc.

Requires: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth for BigQuery.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _load_dotenv() -> None:
    env_file = REPO_ROOT / "data" / "local" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().replace("export ", "", 1).strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        # Do not let empty shell placeholders (e.g. `export BQ_DATASETS_INCLUDE=`) block values from .env.
        prev = os.environ.get(key, "")
        if not (prev or "").strip():
            os.environ[key] = value


def get_datasets_config() -> dict[str, str]:
    """
    Return logical-name -> BQ dataset id.

    Modes:
    - If BQ_DATASETS_INCLUDE is set (comma-separated), use exactly those dataset ids.
    - Otherwise fall back to the traditional 4-layer config (landing/bronze/silver/gold).

    Exclusions:
    - If BQ_DATASETS_EXCLUDE is set (comma-separated), drop those dataset ids.
    """
    include_raw = os.environ.get("BQ_DATASETS_INCLUDE", "").strip()
    exclude_raw = os.environ.get("BQ_DATASETS_EXCLUDE", "").strip()

    exclude = {d.strip() for d in exclude_raw.split(",") if d.strip()}

    if include_raw:
        include = [d.strip() for d in include_raw.split(",") if d.strip()]
        return {d: d for d in include if d not in exclude}

    # Logical layer keys match dbt source names / models/<folder> (BQ dataset ids, not legacy bronze/silver/gold).
    default = {
        "landing": os.environ.get("BQ_DATASET_LANDING", "landing"),
        "raw_dev": os.environ.get("BQ_DATASET_BRONZE", "raw_dev"),
        "staging_dev": os.environ.get("BQ_DATASET_SILVER", "staging_dev"),
        "intermediate_dev": os.environ.get("BQ_DATASET_INTERMEDIATE", "intermediate_dev"),
        "mart_dev": os.environ.get("BQ_DATASET_GOLD", "mart_dev"),
    }
    return {k: v for k, v in default.items() if v and v not in exclude}


def discover_datasets(project: str, client) -> dict[str, str]:
    """
    Auto-discover datasets in the project and return dataset_id -> dataset_id,
    applying BQ_DATASETS_EXCLUDE if present.
    """
    exclude_raw = os.environ.get("BQ_DATASETS_EXCLUDE", "").strip()
    exclude = {d.strip() for d in exclude_raw.split(",") if d.strip()}

    out: dict[str, str] = {}
    try:
        for ds in client.list_datasets(project=project):
            dataset_id = getattr(ds, "dataset_id", None) or getattr(ds, "dataset_id", "")
            if not dataset_id or dataset_id in exclude:
                continue
            out[str(dataset_id)] = str(dataset_id)
    except Exception as e:
        print(f"Dataset discovery failed: {e}", file=sys.stderr)
    return out


def schema_field_to_dict(field) -> dict:
    """Convert BigQuery SchemaField to a simple dict (name, type, mode)."""
    return {
        "name": getattr(field, "name", str(field)),
        "type": getattr(field, "field_type", "STRING"),
        "mode": getattr(field, "mode", "NULLABLE") or "NULLABLE",
    }


def build_catalog(project: str, datasets_config: dict[str, str], client) -> dict:
    """Build catalog: project, datasets (layer -> list of { table, columns })."""
    from google.cloud.bigquery import DatasetReference, TableReference

    catalog: dict = {
        "project": project,
        "datasets": {},
    }
    for layer, dataset_id in datasets_config.items():
        if not dataset_id:
            continue
        dataset_ref = DatasetReference(project, dataset_id)
        tables: list[dict] = []
        try:
            table_list = client.list_tables(dataset_ref)
        except Exception as e:
            print(f"Skip dataset {dataset_id}: {e}", file=sys.stderr)
            catalog["datasets"][layer] = {"dataset": dataset_id, "tables": [], "error": str(e)}
            continue
        for t in table_list:
            table_id = t.table_id
            try:
                table_ref = TableReference(dataset_ref, table_id)
                table = client.get_table(table_ref)
                columns = [schema_field_to_dict(f) for f in table.schema]
                tables.append({"table": table_id, "columns": columns})
            except Exception as e:
                print(f"Skip {dataset_id}.{table_id}: {e}", file=sys.stderr)
        catalog["datasets"][layer] = {"dataset": dataset_id, "tables": tables}
    return catalog


def write_markdown(catalog: dict, out_path: Path) -> None:
    """Write human-readable BIGQUERY_SCHEMA.md."""
    lines = [
        "# BigQuery schema catalog",
        "",
        f"Project: **{catalog['project']}**",
        "",
        "Generated by `data/local/scripts/bq_schema_catalog.py`. Use this to align dbt sources, local DDL, and ingestion.",
        "",
        "---",
        "",
    ]
    for layer, data in catalog["datasets"].items():
        dataset_id = data.get("dataset", layer)
        tables = data.get("tables", [])
        err = data.get("error")
        lines.append(f"## {layer} (`{dataset_id}`)")
        lines.append("")
        if err:
            lines.append(f"*Error: {err}*")
            lines.append("")
            continue
        if not tables:
            lines.append("*No tables.*")
            lines.append("")
            continue
        for t in tables:
            lines.append(f"### `{t['table']}`")
            lines.append("")
            lines.append("| Column | Type | Mode |")
            lines.append("|--------|------|------|")
            for c in t["columns"]:
                lines.append(f"| {c['name']} | {c['type']} | {c.get('mode', 'NULLABLE')} |")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Dump BQ schema for landing/bronze/silver/gold to docs/")
    parser.add_argument("--json-only", action="store_true", help="Write only bq_schema_catalog.json")
    parser.add_argument("--md-only", action="store_true", help="Write only BIGQUERY_SCHEMA.md")
    parser.add_argument("--all-datasets", action="store_true", help="Discover all datasets in the project (respects BQ_DATASETS_EXCLUDE)")
    args = parser.parse_args()

    _load_dotenv()
    project = os.environ.get("BQ_PROJECT", "").strip()
    if not project:
        print("Set BQ_PROJECT (e.g. in data/local/.env)", file=sys.stderr)
        sys.exit(1)

    try:
        from google.cloud import bigquery
    except ImportError:
        print("Install: pip install google-cloud-bigquery", file=sys.stderr)
        sys.exit(1)

    client = bigquery.Client(project=project)
    datasets_config = discover_datasets(project, client) if args.all_datasets else get_datasets_config()
    catalog = build_catalog(project, datasets_config, client)

    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    write_json = not args.md_only
    write_md = not args.json_only

    if write_json:
        json_path = docs_dir / "bq_schema_catalog.json"
        json_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    if write_md:
        md_path = docs_dir / "BIGQUERY_SCHEMA.md"
        write_markdown(catalog, md_path)
        print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
