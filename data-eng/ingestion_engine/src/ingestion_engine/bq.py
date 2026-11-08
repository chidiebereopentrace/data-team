"""Load prepared JSONL into BigQuery landing_dev and append the weekly report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

STRING_EXCEPTIONS = {
    "year": "INT64",
    "value_num": "FLOAT64",
    "latitude": "FLOAT64",
    "longitude": "FLOAT64",
}


def landing_destination(project: str, dataset: str, table: str, mode: str, ingest_date: str) -> tuple[str, str]:
    """Return the BigQuery destination and write disposition for one landing table."""
    base = f"{project}.{dataset}.{table}"
    if mode == "replace_partition":
        stamp = ingest_date.replace("-", "")
        return f"{base}${stamp}", "WRITE_TRUNCATE"
    if mode == "append_snapshot":
        return base, "WRITE_APPEND"
    if mode == "replace_table":
        return base, "WRITE_TRUNCATE"
    raise ValueError(f"unknown load mode: {mode}")


def _schema(row: dict[str, object]) -> list[bigquery.SchemaField]:
    fields = []
    for name in row:
        kind = STRING_EXCEPTIONS.get(name, "STRING")
        if name == "_ingest_date":
            kind = "DATE"
        fields.append(bigquery.SchemaField(name, kind))
    return fields


def _ensure(client: bigquery.Client, table_id: str, schema: list[bigquery.SchemaField]) -> None:
    try:
        client.get_table(table_id)
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="_ingest_date")
        client.create_table(table)


def load_jsonl(
    client: bigquery.Client,
    *,
    project: str,
    dataset: str,
    table: str,
    path: Path,
    mode: str,
    ingest_date: str,
) -> int:
    with path.open(encoding="utf-8") as handle:
        first_line = next((line for line in handle if line.strip()), "")
    if not first_line:
        return 0
    first = json.loads(first_line)
    destination, disposition = landing_destination(project, dataset, table, mode, ingest_date)
    bare = destination.split("$", 1)[0]
    schema = _schema(first)
    _ensure(client, bare, schema)
    with path.open("rb") as handle:
        job = client.load_table_from_file(
            handle,
            destination,
            job_config=bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                schema=schema,
                write_disposition=disposition,
            ),
        )
        job.result()
    return int(job.output_rows or 0)


REPORT_INTEGERS = frozenset({
    "year",
    "manifest_files",
    "sources_enabled",
    "sources_ingested",
    "sources_loaded",
    "sources_partial",
    "sources_failed",
    "sources_skipped",
    "rows_loaded",
    "rows_quarantined",
    "rows_outside_africa",
    "sources_already_done",
})


def run_history(client: bigquery.Client, table_id: str) -> list[dict[str, object]]:
    """Return earlier runs. A missing table means this run is the first."""
    query = f"SELECT run_id, started_at, rows_loaded FROM `{table_id}` ORDER BY started_at"
    try:
        rows = client.query(query).result()
    except NotFound:
        return []
    return [
        {
            "run_id": str(row.run_id or ""),
            "started_at": "" if row.started_at is None else str(row.started_at),
            "rows_loaded": 0 if row.rows_loaded is None else int(row.rows_loaded),
        }
        for row in rows
        if row.run_id
    ]


def attempted_source_ids(client: bigquery.Client, table_id: str) -> set[str]:
    """Return source ids that have already been fetched. A missing table means none."""
    finished, _started = source_cursor(client, table_id)
    return finished


def source_cursor(client: bigquery.Client, table_id: str) -> tuple[set[str], set[str]]:
    """Return finished source ids, and ids left STARTED with no finished row."""
    query = (
        f"SELECT source_id, status FROM `{table_id}` "
        "WHERE status IN ('LOADED', 'PARTIAL', 'FAILED', 'STARTED')"
    )
    try:
        rows = client.query(query).result()
    except NotFound:
        return set(), set()
    finished: set[str] = set()
    started: set[str] = set()
    for row in rows:
        source_id = str(row.source_id or "")
        if not source_id:
            continue
        status = str(row.status or "")
        if status in {"LOADED", "PARTIAL", "FAILED"}:
            finished.add(source_id)
        elif status == "STARTED":
            started.add(source_id)
    return finished, started - finished


def report_load_config(fieldnames: list[str]) -> bigquery.LoadJobConfig:
    """Build an append job that adds a new report column to an existing table."""
    schema = [
        bigquery.SchemaField(name, "INT64" if name in REPORT_INTEGERS else "STRING")
        for name in fieldnames
    ]
    return bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition="WRITE_APPEND",
        autodetect=False,
        schema=schema,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )


def _report_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    return [name.strip() for name in header if name.strip()]


def append_report(client: bigquery.Client, table_id: str, path: Path) -> None:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return
    fieldnames = _report_header(path)
    if not fieldnames:
        return
    with path.open("rb") as handle:
        job = client.load_table_from_file(
            handle,
            table_id,
            job_config=report_load_config(fieldnames),
        )
        job.result()
