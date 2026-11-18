#!/usr/bin/env python3
"""
Fetch BigQuery table schemas for configured transform datasets in BigQuery and generate PostgreSQL DDL.
Writes one .sql file per table to data/local/schema/{landing,raw_dev,staging_dev,intermediate_dev,mart_dev}/ and optionally
creates the tables on the local DB when LOCAL_DB_URL is set.

Usage:
  Set BQ_PROJECT; optionally BQ_DATASET_BRONZE, BQ_DATASET_SILVER, BQ_DATASET_GOLD, LOCAL_DB_URL.
  Tables/views in every configured dataset are discovered live from BigQuery.
  BQ_{LAYER}_TABLES can explicitly restrict a layer for a targeted retry.
  python bq_schema_to_local_pg.py [--write-only] [--execute-only]
  --write-only: only write .sql files (default: write + execute if LOCAL_DB_URL set)
  --execute-only: only execute DDL on local DB (skip writing files; runs all existing .sql in schema/*/)

Requires: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth for BigQuery.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from bq_table_lists import LAYERS, dataset_id_for_layer, load_layer_tables, local_schema_for_layer, local_table_name
from bq_pg_types import postgres_type
from sqlalchemy.dialects import postgresql

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = REPO_ROOT / "data" / "local" / "schema"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from engine_connector import get_engine  # noqa: E402

def bq_type_to_pg(field_type: str, mode: str | None) -> str:
    """Map BigQuery SchemaField type and mode to PostgreSQL type."""
    return str(postgres_type(field_type, mode).compile(dialect=postgresql.dialect()))


def quote_ident(name: str) -> str:
    """Quote all identifiers, preserving BigQuery casing and reserved words."""
    return '"' + str(name).replace('"', '""') + '"'


def safe_filename(table_id: str) -> str:
    """Safe filesystem name for table (e.g. 'new dataset' -> 'new_dataset')."""
    return re.sub(r"[^\w\-]", "_", table_id).strip("_") or "table"


def build_create_table(*, layer: str, table_id: str, schema_fields: list) -> str:
    """Build CREATE SCHEMA + CREATE TABLE IF NOT EXISTS ... (col defs) for PostgreSQL."""
    lines = []
    for f in schema_fields:
        name = getattr(f, "name", str(f))
        field_type = getattr(f, "field_type", "STRING")
        mode = getattr(f, "mode", None)
        pg_type = bq_type_to_pg(field_type, mode)
        col_def = f"  {quote_ident(name)} {pg_type}"
        lines.append(col_def)
    cols = ",\n".join(lines)
    schema_name = quote_ident(local_schema_for_layer(layer))
    table_name = quote_ident(local_table_name(table_id))
    return (
        f"CREATE SCHEMA IF NOT EXISTS {schema_name};\n"
        f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (\n{cols}\n);"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate PostgreSQL DDL from BigQuery schemas (landing/raw_dev/staging_dev/intermediate_dev/mart_dev)")
    parser.add_argument("--write-only", action="store_true", help="Only write .sql files")
    parser.add_argument("--execute-only", action="store_true", help="Only execute DDL on local DB (requires existing .sql)")
    parser.add_argument('--dataset', choices=['all', *LAYERS], default='all', help='Layer to prepare; default: all')
    args = parser.parse_args()
    selected_layers = list(LAYERS) if args.dataset == 'all' else [args.dataset]

    project = os.environ.get("BQ_PROJECT", "").strip()
    local_db_url = os.environ.get("LOCAL_DB_URL", "").strip()
    if not project:
        print("Set BQ_PROJECT (e.g. in .env)", file=sys.stderr)
        sys.exit(1)

    write_files = not args.execute_only
    execute_ddl = (not args.write_only) and bool(local_db_url)

    engine = None
    if execute_ddl and local_db_url:
        engine = get_engine(local_db_url=local_db_url)
    if execute_ddl and engine is None:
        execute_ddl = False

    if args.execute_only:
        if not execute_ddl or engine is None:
            print("--execute-only requires LOCAL_DB_URL", file=sys.stderr)
            sys.exit(1)
        from sqlalchemy import text
        failures = []
        for layer in selected_layers:
            schema_layer_dir = SCHEMA_DIR / layer
            if not schema_layer_dir.exists():
                continue
            for sql_path in sorted(schema_layer_dir.glob("*.sql")):
                ddl = sql_path.read_text(encoding="utf-8")
                try:
                    with engine.begin() as conn:
                        conn.execute(text(ddl))
                    print(f"Executed {layer}/{sql_path.name}")
                except Exception as e:
                    failures.append(f"{layer}/{sql_path.name}: {e}")
                    print(f"Skip failed DDL {layer}/{sql_path.name}: {e}", file=sys.stderr)
        if failures:
            print(f"DDL execution finished with {len(failures)} failure(s).", file=sys.stderr)
            raise SystemExit(1)
        print("Done.")
        return

    try:
        from google.cloud import bigquery
    except ImportError as e:
        print("Install: pip install google-cloud-bigquery", file=sys.stderr)
        raise SystemExit(1) from e

    from google.cloud.bigquery import DatasetReference, TableReference
    client = bigquery.Client(project=project)

    any_tables = False
    failures = []
    metadata_timeout = int(os.environ.get("BQ_METADATA_TIMEOUT_SECONDS", "15"))
    for layer in selected_layers:
        dataset_id = dataset_id_for_layer(layer)

        table_ids = load_layer_tables(layer, client=client)
        if not table_ids:
            continue
        any_tables = True
        schema_layer_dir = SCHEMA_DIR / layer
        schema_layer_dir.mkdir(parents=True, exist_ok=True)
        dataset_ref = DatasetReference(project, dataset_id)
        for table_id in table_ids:
            table_id = table_id.strip()
            if not table_id:
                continue
            try:
                table_ref = TableReference(dataset_ref, table_id)
                table = client.get_table(table_ref, timeout=metadata_timeout)
            except Exception as e:
                print(f"Skip {layer}.{table_id}: {e}", file=sys.stderr)
                failures.append(f"{layer}.{table_id}: {e}")
                continue
            schema_fields = list(table.schema)
            if not schema_fields:
                print(f'SKIP: {dataset_id}.{table_id} has no queryable schema.', flush=True)
                continue
            ddl = build_create_table(layer=layer, table_id=table_id, schema_fields=schema_fields)

            if write_files:
                fname = safe_filename(table_id) + ".sql"
                out_path = schema_layer_dir / fname
                out_path.write_text(ddl + "\n", encoding="utf-8")
                print(f"Wrote {out_path.relative_to(REPO_ROOT)}")

            if execute_ddl and engine is not None:
                from sqlalchemy import text
                try:
                    with engine.begin() as conn:
                        conn.execute(text(ddl))
                    # Local schema matches real BQ dataset id (raw_dev/staging_dev/mart_dev)
                    local_schema = local_schema_for_layer(layer)
                    print(f"Created table {quote_ident(local_schema)}.{quote_ident(table_id)} on local DB")
                except Exception as e:
                    failures.append(f"DDL {layer}.{table_id}: {e}")
                    print(f"Skip failed DDL {layer}.{table_id}: {e}", file=sys.stderr)

    if not any_tables:
        print(
            "No tables found. Check the configured BigQuery datasets and discovery permissions.",
            file=sys.stderr,
        )
    if failures:
        print(f"Schema sync finished with {len(failures)} failure(s); successful DDL was kept.", file=sys.stderr)
        raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()
