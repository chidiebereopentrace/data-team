#!/usr/bin/env python3
"""Profile every column on mart_dev tables; emit per-table RAG YAML + BQ audit table.

Outputs:
  - ml-eng/ml/rag/bq_mart_tables_yaml_files/{table}.yml (one per mart table)
  - {project}.{mart}.audit_mart_column_labels (BigQuery)
  - data-eng/docs/_mart_column_labels_raw.md

Usage (from repo root):
  python data-eng/data/local/scripts/regenerate_mart_table_yamls.py

Requires: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth; BQ_PROJECT in data/local/.env.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DATA_ENG_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = DATA_ENG_ROOT.parent
TABLES_FILE = DATA_ENG_ROOT / "data" / "local" / "scripts" / "mart_dev_tables.txt"
ENTITY_SEED = DATA_ENG_ROOT / "docs" / "mart_entity_dictionary_seed.yaml"
MD_OUT = DATA_ENG_ROOT / "docs" / "_mart_column_labels_raw.md"
YAML_DIR = REPO_ROOT / "ml-eng" / "ml" / "rag" / "bq_mart_tables_yaml_files"
AUDIT_TABLE = "audit_mart_column_labels"
MAX_LABELS = 500
SKIP_TABLES = {"gold_example", AUDIT_TABLE, "audit_mart_ontology_vocab"}

# Preserved on regen (curated by patch_mart_yaml_semantics.py)
CURATED_YAML_KEYS = frozenset(
    {
        "semantic_role",
        "indicator_classes",
        "indicator_families",
        "business_questions_supported",
        "filtering_guidance",
        "sql_generation_hints",
        "semantic_relationships",
        "relationships",
    }
)

COMPLEX_TYPES = frozenset({"GEOGRAPHY", "ARRAY", "STRUCT", "JSON"})
NUMERIC_TYPES = frozenset({"INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC", "INTEGER", "FLOAT"})


def _load_dotenv() -> None:
    env_file = DATA_ENG_ROOT / "data" / "local" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().replace("export ", "", 1).strip()
        value = value.strip().strip("'\"")
        if key and not (os.environ.get(key) or "").strip():
            os.environ[key] = value


def _load_table_list() -> list[str]:
    names: list[str] = []
    for line in TABLES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split()[0])
    return names


def _load_entity_metadata() -> dict[str, dict[str, Any]]:
    if not ENTITY_SEED.is_file():
        return {}
    data = yaml.safe_load(ENTITY_SEED.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in data.get("entities") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("table_name") or "").strip()
        if name:
            out[name] = entry
    return out


def _load_acf_descriptions() -> dict[str, str]:
    if not ENTITY_SEED.is_file():
        return {}
    data = yaml.safe_load(ENTITY_SEED.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for entry in data.get("acf_contract") or []:
        if not isinstance(entry, dict):
            continue
        col = str(entry.get("column") or "").strip()
        meaning = str(entry.get("meaning") or "").strip()
        if col and meaning:
            out[col] = meaning
    return out


@dataclass
class ColumnMeta:
    table_name: str
    column_name: str
    data_type: str


@dataclass
class ColumnProfile:
    table_name: str
    column_name: str
    data_type: str
    profile_mode: str
    distinct_count: int
    null_count: int
    total_rows: int
    is_truncated: bool
    labels: list[str]
    min_value: str | None
    max_value: str | None
    label_rows: list[tuple[str, int, float | None]]
    profiled_at: str


@dataclass
class AuditRow:
    table_name: str
    column_name: str
    data_type: str
    profile_mode: str
    label_value: str | None
    row_count: int | None
    distinct_count: int | None
    null_count: int | None
    pct_of_non_null: float | None
    is_truncated: bool
    profiled_at: str


def _fq(project: str, dataset: str, table: str) -> str:
    return f"`{project}.{dataset}.{table}`"


def _safe_ident(name: str) -> bool:
    return name.replace("_", "").isalnum()


def _is_complex_type(data_type: str) -> bool:
    base = data_type.upper()
    return base in COMPLEX_TYPES or base.startswith("ARRAY") or base.startswith("STRUCT")


def _fetch_live_tables(client, project: str, dataset: str, candidates: list[str]) -> list[str]:
    sql = f"""
    SELECT table_name
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLES`
    WHERE table_type = 'BASE TABLE'
    """
    live = {r.table_name for r in client.query(sql).result()}
    return [t for t in candidates if t in live and t not in SKIP_TABLES]


def _fetch_columns(client, project: str, dataset: str, table_name: str) -> list[ColumnMeta]:
    from google.cloud import bigquery

    sql = f"""
    SELECT column_name, data_type
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @table_name
    ORDER BY ordinal_position
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("table_name", "STRING", table_name)]
    )
    rows = client.query(sql, job_config=job_config).result()
    return [ColumnMeta(table_name, r.column_name, r.data_type) for r in rows]


def _column_stats(client, project: str, dataset: str, col: ColumnMeta) -> tuple[int, int, int]:
    if not _safe_ident(col.column_name):
        return 0, 0, 0
    if _is_complex_type(col.data_type):
        sql = f"""
        SELECT
          0 AS distinct_count,
          COUNTIF({col.column_name} IS NULL) AS null_count,
          COUNT(*) AS total_rows
        FROM {_fq(project, dataset, col.table_name)}
        """
    else:
        sql = f"""
        SELECT
          APPROX_COUNT_DISTINCT({col.column_name}) AS distinct_count,
          COUNTIF({col.column_name} IS NULL) AS null_count,
          COUNT(*) AS total_rows
        FROM {_fq(project, dataset, col.table_name)}
        """
    row = next(iter(client.query(sql).result()))
    return int(row.distinct_count or 0), int(row.null_count or 0), int(row.total_rows or 0)


def _column_labels(
    client,
    project: str,
    dataset: str,
    col: ColumnMeta,
    limit: int,
) -> list[tuple[str, int]]:
    if not _safe_ident(col.column_name):
        return []
    sql = f"""
    SELECT
      CAST({col.column_name} AS STRING) AS label_value,
      COUNT(*) AS row_count
    FROM {_fq(project, dataset, col.table_name)}
    WHERE {col.column_name} IS NOT NULL
    GROUP BY 1
    ORDER BY row_count DESC
    LIMIT {limit}
    """
    return [(r.label_value, int(r.row_count)) for r in client.query(sql).result()]


def _numeric_bounds(client, project: str, dataset: str, col: ColumnMeta) -> tuple[str | None, str | None]:
    if not _safe_ident(col.column_name):
        return None, None
    sql = f"""
    SELECT
      CAST(MIN({col.column_name}) AS STRING) AS min_value,
      CAST(MAX({col.column_name}) AS STRING) AS max_value
    FROM {_fq(project, dataset, col.table_name)}
    WHERE {col.column_name} IS NOT NULL
    """
    row = next(iter(client.query(sql).result()))
    return row.min_value, row.max_value


def profile_column(
    client,
    project: str,
    dataset: str,
    col: ColumnMeta,
    profiled_at: str,
) -> ColumnProfile:
    distinct_count, null_count, total_rows = _column_stats(client, project, dataset, col)
    non_null = max(total_rows - null_count, 0)
    use_labels = not _is_complex_type(col.data_type) and distinct_count <= MAX_LABELS

    if use_labels:
        raw_labels = _column_labels(client, project, dataset, col, MAX_LABELS)
        label_rows: list[tuple[str, int, float | None]] = []
        labels: list[str] = []
        for label_value, count in raw_labels:
            pct = (100.0 * count / non_null) if non_null else None
            label_rows.append((label_value, count, round(pct, 4) if pct is not None else None))
            labels.append(label_value)
        return ColumnProfile(
            table_name=col.table_name,
            column_name=col.column_name,
            data_type=col.data_type,
            profile_mode="labels",
            distinct_count=distinct_count,
            null_count=null_count,
            total_rows=total_rows,
            is_truncated=distinct_count > MAX_LABELS,
            labels=labels,
            min_value=None,
            max_value=None,
            label_rows=label_rows,
            profiled_at=profiled_at,
        )

    min_val, max_val = _numeric_bounds(client, project, dataset, col)
    return ColumnProfile(
        table_name=col.table_name,
        column_name=col.column_name,
        data_type=col.data_type,
        profile_mode="stats",
        distinct_count=distinct_count,
        null_count=null_count,
        total_rows=total_rows,
        is_truncated=distinct_count > MAX_LABELS,
        labels=[],
        min_value=min_val,
        max_value=max_val,
        label_rows=[],
        profiled_at=profiled_at,
    )


def _column_description(column_name: str, acf_desc: dict[str, str]) -> str:
    if column_name in acf_desc:
        return acf_desc[column_name]
    return column_name.replace("_", " ")


def _build_table_yaml(
    table_name: str,
    columns: list[ColumnMeta],
    profiles: list[ColumnProfile],
    project: str,
    dataset: str,
    profiled_at: str,
    entity_meta: dict[str, dict[str, Any]],
    acf_desc: dict[str, str],
) -> dict[str, Any]:
    meta = entity_meta.get(table_name) or {}
    payload: dict[str, Any] = {
        "table_name": f"{project}.{dataset}.{table_name}",
        "description": str(meta.get("purpose") or meta.get("analytical_use") or table_name.replace("_", " ")),
        "source": {"layer": "mart_dev"},
        "profiled_at": profiled_at,
    }
    if meta.get("grain"):
        payload["grain"] = meta["grain"]
    if meta.get("domain"):
        payload["entity_type"] = meta["domain"]

    payload["columns"] = [
        {
            "name": col.column_name,
            "type": col.data_type,
            "description": _column_description(col.column_name, acf_desc),
        }
        for col in columns
    ]

    for prof in profiles:
        if prof.profile_mode == "labels" and prof.labels:
            payload[f"{prof.column_name}_value_samples"] = prof.labels
        else:
            stats: dict[str, Any] = {
                "distinct_count": prof.distinct_count,
                "null_count": prof.null_count,
                "is_truncated": prof.is_truncated,
            }
            if prof.min_value is not None:
                stats["min_value"] = prof.min_value
            if prof.max_value is not None:
                stats["max_value"] = prof.max_value
            payload[f"{prof.column_name}_value_stats"] = stats

    existing_path = YAML_DIR / f"{table_name}.yml"
    if existing_path.is_file():
        try:
            existing = yaml.safe_load(existing_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                for key in CURATED_YAML_KEYS:
                    if key in existing and existing[key]:
                        payload[key] = existing[key]
        except (OSError, yaml.YAMLError):
            pass

    return payload


def _profiles_to_audit_rows(profiles: list[ColumnProfile]) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for prof in profiles:
        if prof.profile_mode == "labels":
            non_null = max(prof.total_rows - prof.null_count, 0)
            if not prof.label_rows:
                rows.append(
                    AuditRow(
                        table_name=prof.table_name,
                        column_name=prof.column_name,
                        data_type=prof.data_type,
                        profile_mode="labels",
                        label_value=None,
                        row_count=0,
                        distinct_count=prof.distinct_count,
                        null_count=prof.null_count,
                        pct_of_non_null=None,
                        is_truncated=prof.is_truncated,
                        profiled_at=prof.profiled_at,
                    )
                )
            for label_value, count, pct in prof.label_rows:
                rows.append(
                    AuditRow(
                        table_name=prof.table_name,
                        column_name=prof.column_name,
                        data_type=prof.data_type,
                        profile_mode="labels",
                        label_value=label_value,
                        row_count=count,
                        distinct_count=prof.distinct_count,
                        null_count=prof.null_count,
                        pct_of_non_null=pct,
                        is_truncated=prof.is_truncated,
                        profiled_at=prof.profiled_at,
                    )
                )
        else:
            note = None
            if prof.min_value is not None or prof.max_value is not None:
                note = f"min={prof.min_value};max={prof.max_value}"
            rows.append(
                AuditRow(
                    table_name=prof.table_name,
                    column_name=prof.column_name,
                    data_type=prof.data_type,
                    profile_mode="stats",
                    label_value=note,
                    row_count=prof.total_rows,
                    distinct_count=prof.distinct_count,
                    null_count=prof.null_count,
                    pct_of_non_null=None,
                    is_truncated=prof.is_truncated,
                    profiled_at=prof.profiled_at,
                )
            )
    return rows


def _audit_rows_to_bq_dicts(rows: list[AuditRow]) -> list[dict[str, Any]]:
    return [
        {
            "table_name": r.table_name,
            "column_name": r.column_name,
            "data_type": r.data_type,
            "profile_mode": r.profile_mode,
            "label_value": r.label_value,
            "row_count": r.row_count,
            "distinct_count": r.distinct_count,
            "null_count": r.null_count,
            "pct_of_non_null": r.pct_of_non_null,
            "is_truncated": r.is_truncated,
            "profiled_at": r.profiled_at,
        }
        for r in rows
    ]


def _load_bq_table(client, project: str, dataset: str, rows: list[dict[str, Any]]) -> None:
    from google.cloud import bigquery

    table_id = f"{project}.{dataset}.{AUDIT_TABLE}"
    schema = [
        bigquery.SchemaField("table_name", "STRING"),
        bigquery.SchemaField("column_name", "STRING"),
        bigquery.SchemaField("data_type", "STRING"),
        bigquery.SchemaField("profile_mode", "STRING"),
        bigquery.SchemaField("label_value", "STRING"),
        bigquery.SchemaField("row_count", "INT64"),
        bigquery.SchemaField("distinct_count", "INT64"),
        bigquery.SchemaField("null_count", "INT64"),
        bigquery.SchemaField("pct_of_non_null", "FLOAT64"),
        bigquery.SchemaField("is_truncated", "BOOL"),
        bigquery.SchemaField("profiled_at", "TIMESTAMP"),
    ]
    client.delete_table(table_id, not_found_ok=True)
    client.create_table(bigquery.Table(table_id, schema=schema))
    if not rows:
        print(f"No rows to load into {table_id}")
        return
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    client.load_table_from_json(rows, table_id, job_config=job_config).result()
    print(f"Loaded {len(rows):,} rows to {table_id}")


def _write_markdown(
    by_table: dict[str, list[ColumnProfile]],
    project: str,
    dataset: str,
) -> None:
    lines = [
        f"> **Snapshot note:** Regenerated from `{project}.{dataset}` ({date.today().isoformat()}). "
        "See [MART_QA_NOTES.md](./MART_QA_NOTES.md).",
        "",
    ]
    for table_name in sorted(by_table.keys()):
        profiles = by_table[table_name]
        lines.extend([f"## {table_name}", ""])
        for prof in profiles:
            if prof.profile_mode == "labels":
                flag = " (truncated)" if prof.is_truncated else ""
                lines.append(
                    f"### {prof.column_name} ({prof.data_type}, distinct={prof.distinct_count}){flag}"
                )
                lines.append("| label_value | n | pct |")
                lines.append("|---|---:|---:|")
                for label_value, count, pct in prof.label_rows[:100]:
                    pct_s = "" if pct is None else f"{pct:.1f}"
                    lines.append(f"| `{label_value}` | {count:,} | {pct_s} |")
                if len(prof.label_rows) > 100:
                    lines.append(f"| … | {len(prof.label_rows) - 100} more | |")
            else:
                min_max = ""
                if prof.min_value is not None or prof.max_value is not None:
                    min_max = f", min={prof.min_value}, max={prof.max_value}"
                lines.append(
                    f"### {prof.column_name} ({prof.data_type}, stats only, "
                    f"distinct={prof.distinct_count}{min_max})"
                )
            lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {MD_OUT}")


def main() -> int:
    _load_dotenv()
    from google.cloud import bigquery

    project = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4")
    dataset = os.environ.get("BQ_DATASET_GOLD", "mart_dev")
    profiled_at = datetime.now(timezone.utc).isoformat()

    entity_meta = _load_entity_metadata()
    acf_desc = _load_acf_descriptions()

    candidates = _load_table_list()
    client = bigquery.Client(project=project)
    tables = _fetch_live_tables(client, project, dataset, candidates)
    if not tables:
        print(f"No live tables found in {project}.{dataset}", file=sys.stderr)
        return 1

    YAML_DIR.mkdir(parents=True, exist_ok=True)
    all_audit: list[AuditRow] = []
    by_table_profiles: dict[str, list[ColumnProfile]] = {}
    yaml_count = 0

    for idx, table_name in enumerate(tables, start=1):
        print(f"[{idx}/{len(tables)}] {table_name}", flush=True)
        try:
            columns = _fetch_columns(client, project, dataset, table_name)
        except Exception as exc:
            print(f"  skip columns: {exc}", file=sys.stderr)
            continue

        profiles: list[ColumnProfile] = []
        for col in columns:
            try:
                profiles.append(profile_column(client, project, dataset, col, profiled_at))
            except Exception as exc:
                print(f"  skip {col.column_name}: {exc}", file=sys.stderr)
                continue

        by_table_profiles[table_name] = profiles
        all_audit.extend(_profiles_to_audit_rows(profiles))

        table_yaml = _build_table_yaml(
            table_name,
            columns,
            profiles,
            project,
            dataset,
            profiled_at,
            entity_meta,
            acf_desc,
        )
        out_path = YAML_DIR / f"{table_name}.yml"
        out_path.write_text(
            yaml.safe_dump(table_yaml, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        yaml_count += 1

    _write_markdown(by_table_profiles, project, dataset)
    _load_bq_table(client, project, dataset, _audit_rows_to_bq_dicts(all_audit))

    summary = {
        "tables_profiled": len(tables),
        "yaml_files_written": yaml_count,
        "audit_rows": len(all_audit),
        "yaml_dir": str(YAML_DIR),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
