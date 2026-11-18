#!/usr/bin/env python3
"""
Sync data from BigQuery into the local database for a given layer (landing, raw_dev, staging_dev, intermediate_dev, or mart_dev).
Discovers current tables/views from BigQuery for every layer. Each run refreshes
the local snapshots using the same configurable row limit for every table.

Usage:
  Set BQ_PROJECT, BQ_DATASET_BRONZE / BQ_DATASET_SILVER / BQ_DATASET_GOLD, LOCAL_DB_URL (or LOCAL_DB_PATH).
  python sync_all_tables.py --dataset raw_dev [--limit N] [--timeout SECONDS]
  python sync_all_tables.py --dataset staging_dev
  python sync_all_tables.py --dataset mart_dev
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bq_table_lists import LAYERS, dataset_id_for_layer, load_layer_tables, local_schema_for_layer, local_table_name

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent
SYNC_SCRIPT = SCRIPTS_DIR / "bq_partition_to_local.py"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Sync BigQuery tables to local DB for a layer (landing/raw_dev/staging_dev/intermediate_dev/mart_dev)")
    parser.add_argument("--dataset", choices=LAYERS, required=True, help="Layer to sync")
    parser.add_argument("--limit", type=int, default=None, help="Override row limits for every table; 0 copies all rows")
    parser.add_argument('--workers', type=int, default=4, help='Concurrent independent table imports (default: 4)')
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-table deadline in seconds; 0 disables it (default for full copies; samples default to 180s)",
    )
    args = parser.parse_args()

    if args.timeout is not None and args.timeout < 0:
        parser.error("--timeout must be zero (no deadline) or positive")
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    if args.limit is not None and args.limit < 0:
        parser.error('--limit must be zero (all rows) or positive')

    if not SYNC_SCRIPT.exists():
        print(f"Sync script not found: {SYNC_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    table_ids = load_layer_tables(args.dataset)
    if not table_ids:
        ds_id = dataset_id_for_layer(args.dataset)
        print(
            f"No tables found for layer={args.dataset} (dataset={ds_id}). "
            f"Check BigQuery discovery or set BQ_{args.dataset.upper()}_TABLES for an explicit subset.",
            file=sys.stderr,
        )
        sys.exit(0)

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env["BQ_DATASET"] = dataset_id_for_layer(args.dataset)
    env["LOCAL_SCHEMA"] = local_schema_for_layer(args.dataset)
    # Ensure DB selection is consistent in subprocesses (avoid falling back to SQLite)
    if "LOCAL_DB_URL" in os.environ:
        env["LOCAL_DB_URL"] = os.environ.get("LOCAL_DB_URL", "")
    if args.limit is not None:
        env["BQ_PARTITION_LIMIT"] = str(args.limit)
    limit = int(env.get('BQ_PARTITION_LIMIT', '0'))
    timeout = args.timeout
    if timeout is None:
        timeout = int(env.get('BQ_TABLE_TIMEOUT_SECONDS', '0' if limit == 0 else '180'))
    if timeout < 0:
        parser.error('BQ_TABLE_TIMEOUT_SECONDS must be zero (no deadline) or positive')

    print(f'Selected {len(table_ids)} tables/views from {env["BQ_DATASET"]}.', flush=True)
    # PostgreSQL truncates long names. Refuse collisions before concurrent writes.
    local_names = [local_table_name(name) for name in table_ids]
    if len(local_names) != len(set(local_names)):
        raise ValueError('Selected table names collide after PostgreSQL identifier truncation')

    def sync_table(table_id):
        cmd = [
            sys.executable,
            str(SYNC_SCRIPT),
            "--table", table_id,
            "--target-table", table_id,
            "--target-schema", local_schema_for_layer(args.dataset),
        ]
        if args.limit is not None:
            cmd.extend(['--limit', str(args.limit)])
        print(f"Syncing {args.dataset}/{table_id} ...", flush=True)
        try:
            r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), timeout=timeout or None)
        except subprocess.TimeoutExpired:
            message = f"Warning: {table_id} exceeded {timeout}s and was skipped"
            print(message, file=sys.stderr)
            return f"{table_id} (timeout)"
        if r.returncode != 0:
            message = f"Warning: {table_id} failed (exit {r.returncode}); moving to next table"
            print(message, file=sys.stderr)
            return f"{table_id} (exit {r.returncode})"
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(sync_table, name): name for name in table_ids}
        failures = []
        for done, future in enumerate(as_completed(pending), start=1):
            failure = future.result()
            if failure:
                failures.append(failure)
            print(f'Tables finished: {done}/{len(table_ids)}; failures: {len(failures)}; '
                  f'last: {pending[future]}', flush=True)

    if failures:
        print(f"Finished {args.dataset} with {len(failures)} failed table(s):", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        sys.exit(1)
    print(f"Finished {args.dataset}: all selected tables synced.")


if __name__ == "__main__":
    main()
