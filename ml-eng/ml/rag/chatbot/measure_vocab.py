"""Distinct measure names for each fact, checked against COUNT(DISTINCT).

Table YAML keeps a short preview. This writes the complete name list into
``value_index/mart_value_index.json`` under ``enums``. A column is written only
when the pulled list has the same length as ``COUNT(DISTINCT)``. Prices stay in
``price_vocab.json``.

Build:
  cd ml-eng && python -m ml.rag.chatbot.measure_vocab --build
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import bigquery

from ml.rag.chatbot.bq_mart_sql import mart_dataset
from ml.rag.chatbot.bq_table_schema_yaml import (
    MEASURE_NAME_COLUMNS,
    columns_for_mart_tables,
    known_mart_table_names,
)
from ml.rag.chatbot.value_index import clear_value_index_cache
from ml.rag.local_env import load_rag_dotenv

INDEX = Path(__file__).resolve().parents[1] / "value_index" / "mart_value_index.json"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BYTES = 2 * 1024 * 1024 * 1024


def accept_names(values: list[str] | None, distinct_count: int) -> list[str] | None:
    """The name list is usable only when it is the whole distinct domain."""
    cleaned = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not cleaned or len(cleaned) != int(distinct_count):
        return None
    return sorted(set(cleaned), key=str.casefold)


def fact_tables() -> list[str]:
    names: set[str] = set()
    for name in known_mart_table_names():
        bare = str(name).split(".")[-1].lower()
        if bare.startswith(("fct_", "agg_")):
            names.add(bare)
    return sorted(names)


def _name_columns(table_id: str) -> list[str]:
    cols = columns_for_mart_tables({table_id}).get(table_id) or set()
    return sorted(col for col in cols if col.lower() in MEASURE_NAME_COLUMNS and _IDENT_RE.match(col))


def _query(project: str, dataset: str, table_id: str, columns: list[str]) -> str:
    pieces: list[str] = []
    for col in columns:
        expr = (
            f"IF({col} IS NULL OR TRIM(CAST({col} AS STRING)) = '', NULL, "
            f"TRIM(CAST({col} AS STRING)))"
        )
        pieces.append(f"ARRAY_AGG(DISTINCT {expr} IGNORE NULLS) AS {col}_vals")
        pieces.append(f"COUNT(DISTINCT {expr}) AS {col}_n")
    return f"SELECT {', '.join(pieces)} FROM `{project}.{dataset}.{table_id}`"


def build_measure_names(client: bigquery.Client) -> dict[str, list[str]]:
    """Pull each fact's name columns. Skip a column whose list does not match the count."""
    project = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4").strip()
    dataset = mart_dataset()
    found: dict[str, list[str]] = {}
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=_BYTES)
    for table_id in fact_tables():
        columns = _name_columns(table_id)
        if not columns or not _IDENT_RE.match(table_id):
            continue
        sql = _query(project, dataset, table_id, columns)
        try:
            rows = list(client.query(sql, job_config=job_config).result(timeout=180))
        except Exception as exc:
            print(f"skip {table_id}: {exc}")
            continue
        if not rows:
            continue
        row = rows[0]
        for col in columns:
            values = list(row[f"{col}_vals"] or [])
            distinct = int(row[f"{col}_n"] or 0)
            accepted = accept_names(values, distinct)
            if accepted:
                found[f"{table_id}.{col}"] = accepted
            elif distinct or values:
                print(f"refuse {table_id}.{col}: list {len(values)} count {distinct}")
    return found


def write_measure_names(names: dict[str, list[str]], path: Path = INDEX) -> Path:
    """Merge verified columns into the value index. Other enum keys stay."""
    current: dict = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            current = loaded
    enums = dict(current.get("enums") or {})
    enums.update(names)
    current["enums"] = enums
    current["measure_names_built_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    clear_value_index_cache()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild measure-name enums from live BigQuery")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return 1
    load_rag_dotenv(Path(__file__).resolve().parents[3])
    client = bigquery.Client(project=os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4"))
    names = build_measure_names(client)
    if not names:
        print("no measure names verified; value index left unchanged")
        return 1
    path = write_measure_names(names)
    print(f"{len(names)} name columns -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INDEX",
    "accept_names",
    "build_measure_names",
    "fact_tables",
    "write_measure_names",
]
