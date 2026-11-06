#!/usr/bin/env python3
"""Rewrite the local-partition table lists from live BigQuery.

Writes data/local/scripts/{layer}_tables.txt for:
  landing, raw_dev, staging_dev, intermediate_dev, mart_dev

Keeps TABLE and EXTERNAL. Skips views, materialized views, and snapshots.
Names longer than 63 bytes are recorded in data/local/out/layer_table_names.csv.

A dataset that cannot be listed is reported and its file is left unchanged.
Write mode exits 0 when at least one layer was refreshed, so the rest of
populate_local_db.sh still loads. --check exits 1 when any list differs or
any dataset cannot be listed.

Usage (from data-eng/):
  python data/local/scripts/sync_layer_table_lists.py
  python data/local/scripts/sync_layer_table_lists.py --check
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from google.cloud import bigquery
from google.cloud.bigquery import DatasetReference

from bq_table_lists import LAYERS, dataset_id_for_layer, local_table_name

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent
OUT_CSV = REPO_ROOT / "data" / "local" / "out" / "layer_table_names.csv"
KEEP_TYPES = {"TABLE", "EXTERNAL"}


def _load_dotenv() -> None:
    env_file = REPO_ROOT / "data" / "local" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().replace("export ", "", 1).strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        prev = os.environ.get(key, "")
        if not (prev or "").strip():
            os.environ[key] = value


def list_base_tables(client: bigquery.Client, project: str, dataset_id: str) -> tuple[list[str], int]:
    kept: list[str] = []
    skipped = 0
    for item in client.list_tables(DatasetReference(project, dataset_id)):
        table_type = str(getattr(item, "table_type", "") or "")
        table_id = str(getattr(item, "table_id", "") or "")
        if not table_id:
            continue
        if table_type not in KEEP_TYPES:
            skipped += 1
            continue
        kept.append(table_id)
    return sorted(kept), skipped


def read_listed_tables(path: Path) -> list[str]:
    if not path.is_file():
        return []
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def render(dataset_id: str, tables: list[str]) -> str:
    lines = [
        f"# Live base tables in {dataset_id}. Written by sync_layer_table_lists.py.",
        "# TABLE and EXTERNAL only. Views are omitted.",
        "",
        *tables,
        "",
    ]
    return "\n".join(lines)


def write_name_index(rows: list[tuple[str, str, str]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "bq_table", "local_table"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the five local-partition table lists from BigQuery")
    parser.add_argument("--check", action="store_true", help="Exit 1 if a list file differs from BigQuery")
    args = parser.parse_args()

    _load_dotenv()
    project = os.environ.get("BQ_PROJECT", "").strip()
    if not project:
        print("Set BQ_PROJECT (data/local/.env)", file=sys.stderr)
        return 1

    client = bigquery.Client(project=project)
    drifted = False
    written = 0
    shortened: list[tuple[str, str, str]] = []
    for layer in LAYERS:
        dataset_id = dataset_id_for_layer(layer)
        path = SCRIPTS_DIR / f"{layer}_tables.txt"
        try:
            tables, skipped = list_base_tables(client, project, dataset_id)
        except Exception as exc:
            print(f"Skip {layer} ({dataset_id}): {exc}", file=sys.stderr)
            drifted = True
            continue
        if args.check:
            current = read_listed_tables(path)
            if current != tables:
                drifted = True
                print(
                    f"{path.name} out of date ({len(current)} listed, {len(tables)} in {dataset_id}, {skipped} skipped)"
                )
            else:
                print(f"{path.name} matches {dataset_id} ({len(tables)} tables)")
            continue
        path.write_text(render(dataset_id, tables), encoding="utf-8")
        written += 1
        for table_id in tables:
            local_name = local_table_name(table_id)
            if local_name != table_id:
                shortened.append((layer, table_id, local_name))
        print(f"Wrote {len(tables)} tables to {path.name} ({skipped} views skipped)")

    if not args.check and written:
        write_name_index(shortened)
        print(f"Wrote {len(shortened)} shortened names to {OUT_CSV.relative_to(REPO_ROOT)}")
    if args.check:
        return 1 if drifted else 0
    if written == 0:
        print("No layer lists were refreshed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
