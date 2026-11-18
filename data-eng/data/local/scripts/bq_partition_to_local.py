#!/usr/bin/env python3
"""
Sync a partition (or sample) from a BigQuery database (dataset) into the local database.
The source is a BigQuery table or view; the destination is local PostgreSQL.
Configure BQ_PROJECT, BQ_DATASET, BQ_TABLE and LOCAL_DB_URL.

Usage:
  PostgreSQL (set LOCAL_DB_URL in .env):
    python bq_partition_to_local.py [--table TABLE] [--partition-filter "date='2024-01-01'"] [--limit N]

Requires: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth for BigQuery.
"""
from __future__ import annotations

import os
import sys
from itertools import islice
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from engine_connector import get_engine  # noqa: E402
from bq_table_lists import local_table_name
from bq_pg_types import postgres_type, value_for_field  # noqa: E402
from copy_progress import CopyProgress  # noqa: E402


def _load_dotenv() -> None:
    """Load data/local/.env into os.environ when not already set (run from repo root)."""
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
        if key and key not in os.environ:
            os.environ[key] = value


def get_config() -> dict:
    return {
        "bq_project": os.environ.get("BQ_PROJECT", ""),
        "bq_dataset": os.environ.get("BQ_DATASET", ""),
        "bq_table": os.environ.get("BQ_TABLE", "bronze_events"),
        "partition_filter": os.environ.get("BQ_PARTITION_FILTER", ""),
        "limit": int(os.environ.get("BQ_PARTITION_LIMIT", "0")),
        "local_db_url": os.environ.get("LOCAL_DB_URL", "").strip(),
        "target_table": os.environ.get("LOCAL_TABLE", "bronze_events"),
        "target_schema": os.environ.get("LOCAL_SCHEMA", "").strip(),
    }


def quote_bq(name: str) -> str:
    return "`" + str(name).replace("`", "") + "`"


def build_query(config: dict) -> str:
    project = config["bq_project"]
    dataset = config["bq_dataset"]
    table = config["bq_table"]
    if not project or not dataset:
        raise ValueError("Set BQ_PROJECT and BQ_DATASET (e.g. in .env)")
    full_table = f"{quote_bq(project)}.{quote_bq(dataset)}.{quote_bq(table)}"
    query = f"SELECT * FROM {full_table}"
    if config["partition_filter"]:
        query += f" WHERE {config['partition_filter']}"
    if config['limit'] < 0:
        raise ValueError('Row limit must be zero (all rows) or positive')
    if config['limit']:
        query += f" LIMIT {config['limit']}"
    return query


def write_rows(rows, fields, conn, *, table: str, schema=None, batch_size=1000, progress=None) -> int:
    """Replace a snapshot in bounded batches without retaining the full source in memory."""
    import pandas as pd
    columns = [field.name for field in fields]
    dtypes = {field.name: postgres_type(field.field_type, field.mode) for field in fields}
    iterator = iter(rows)
    first = True
    copied = 0
    while True:
        if progress:
            progress.update('reading BigQuery rows', copied=copied)
        batch = list(islice(iterator, batch_size))
        if not batch and not first:
            break
        records = [{field.name: value_for_field(row[field.name], field) for field in fields}
                   for row in batch]
        # Keep nullable integers/decimals exact; SQL types come from metadata.
        df = pd.DataFrame(records, columns=columns, dtype=object)
        if progress:
            progress.update('writing PostgreSQL rows', copied=copied)
        df.to_sql(table, conn, if_exists='replace' if first else 'append',
                  index=False, method='multi', chunksize=max(1, min(500, 30000 // max(len(columns), 1))),
                  dtype=dtypes, schema=schema)
        copied += len(batch)
        first = False
        if not batch:
            break
    return copied


def sync_table(config: dict, client=None) -> int:
    """Stream one source into a PostgreSQL snapshot; return committed row count."""
    config = dict(config)
    if not config.get("local_db_url", "").startswith(("postgresql://", "postgresql+")):
        raise ValueError("Set LOCAL_DB_URL to a PostgreSQL connection URL")
    query_timeout = config.get("query_timeout")
    if query_timeout is None:
        query_timeout = int(os.environ.get("BQ_QUERY_TIMEOUT_SECONDS", "0" if config["limit"] == 0 else "45"))
    if query_timeout < 0:
        raise ValueError("Query timeout must be zero (no deadline) or positive")
    if config["local_db_url"].startswith("postgres"):
        original_name = config["target_table"]
        config['target_table'] = local_table_name(original_name)
        if config['target_table'] != original_name:
            print(f"PostgreSQL table mapping: {original_name} -> {config['target_table']}", flush=True)

    from google.cloud import bigquery

    query = build_query(config)
    print("Query:", query[:200], "...", flush=True)

    if client is None:
        client = bigquery.Client(project=config["bq_project"])
    metadata = client.get_table(
        f"{config['bq_project']}.{config['bq_dataset']}.{config['bq_table']}",
        timeout=int(os.environ.get('BQ_METADATA_TIMEOUT_SECONDS', '15')))
    if not metadata.schema:
        print(f"SKIP: {config['bq_dataset']}.{config['bq_table']} has no queryable schema.", flush=True)
        return 0
    job = client.query(query)
    try:
        with CopyProgress(f"{config['bq_dataset']}.{config['bq_table']}") as progress:
            rows = job.result(timeout=query_timeout or None)
            fields = list(rows.schema or [])
            if not fields:
                raise ValueError('Source table has no schema')
            progress.update('opening local transaction', total=rows.total_rows)
            engine = get_engine(config=config)
            with engine.begin() as conn:
                schema = (config.get('target_schema') or '').strip() or None
                if schema:
                    from sqlalchemy import text
                    escaped = schema.replace('"', '""')
                    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{escaped}"'))
                copied = write_rows(rows, fields, conn, table=config['target_table'], schema=schema,
                                    progress=progress)
                progress.update('committing local transaction', copied=copied)
            progress.update('committed', copied=copied)
    except BaseException:
        try:
            job.cancel()
        except Exception:
            pass
        raise

    target = str(engine.url)
    full_target = f"{schema}.{config['target_table']}" if schema else config["target_table"]
    print(f"Wrote {copied} rows to {target} table {full_target}", flush=True)

    return copied


def main() -> None:
    import argparse

    _load_dotenv()

    parser = argparse.ArgumentParser(description="Sync a BigQuery table or view to local PostgreSQL")
    parser.add_argument("--table", default=os.environ.get("BQ_TABLE", "bronze_events"))
    parser.add_argument("--partition-filter", default=os.environ.get("BQ_PARTITION_FILTER", ""))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("BQ_PARTITION_LIMIT", "0")), help='Maximum rows; default 0 copies all rows')
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=None,
        help="Query deadline in seconds; 0 disables it (default for full copies; samples default to 45s)",
    )
    parser.add_argument("--target-table", default=os.environ.get("LOCAL_TABLE", "bronze_events"))
    parser.add_argument("--target-schema", default=os.environ.get("LOCAL_SCHEMA", ""), help="Postgres schema to write into")
    args = parser.parse_args()

    config = get_config()
    if args.table:
        config["bq_table"] = args.table
    if args.partition_filter:
        config["partition_filter"] = args.partition_filter
    if args.limit < 0:
        parser.error('--limit must be zero (all rows) or positive')
    config["limit"] = args.limit
    query_timeout = args.query_timeout
    if query_timeout is None:
        query_timeout = int(os.environ.get('BQ_QUERY_TIMEOUT_SECONDS', '0' if args.limit == 0 else '45'))
    if query_timeout < 0:
        parser.error("--query-timeout must be zero (no deadline) or positive")
    if args.target_table:
        config["target_table"] = args.target_table
    if args.target_schema:
        config["target_schema"] = args.target_schema
    config["query_timeout"] = query_timeout
    try:
        sync_table(config)
    except Exception as exc:
        print(f"Snapshot copy failed for {config['bq_dataset']}.{config['bq_table']}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
