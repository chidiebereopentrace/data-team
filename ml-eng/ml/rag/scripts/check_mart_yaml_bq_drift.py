#!/usr/bin/env python3
"""Compare allowlisted mart YAML column sets vs live BigQuery INFORMATION_SCHEMA.

Fails when a YAML file's columns diverge from BQ (add/drop/rename) for tables listed
in data-eng/data/local/scripts/mart_dev_rag_tables.txt that also have a YAML file.

PR CI stays offline (mart YAML contract tests). This job is scheduled / secrets-required.

Usage (from repo root):
  PYTHONPATH=ml-eng python ml-eng/ml/rag/scripts/check_mart_yaml_bq_drift.py
  PYTHONPATH=ml-eng python ml-eng/ml/rag/scripts/check_mart_yaml_bq_drift.py --tables fct_production,dim_geography

Requires: GOOGLE_APPLICATION_CREDENTIALS or gcloud ADC; BQ_PROJECT (and optional BQ_DATASET_GOLD).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
TABLES_FILE = REPO_ROOT / "data-eng" / "data" / "local" / "scripts" / "mart_dev_rag_tables.txt"
YAML_DIR = REPO_ROOT / "ml-eng" / "ml" / "rag" / "bq_mart_tables_yaml_files"
SKIP_TABLES = frozenset({"gold_example", "audit_mart_column_labels", "audit_mart_ontology_vocab"})


def _load_dotenv() -> None:
    for rel in ("ml-eng/config/.env", "data-eng/data/local/.env", ".env"):
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _allowlisted_tables(only: list[str] | None) -> list[str]:
    if only:
        return [t.strip() for t in only if t.strip() and t.strip() not in SKIP_TABLES]
    if not TABLES_FILE.is_file():
        raise SystemExit(f"missing tables list: {TABLES_FILE}")
    out: list[str] = []
    for line in TABLES_FILE.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#") or t in SKIP_TABLES:
            continue
        out.append(t)
    return out


def _yaml_columns(path: Path) -> set[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cols = raw.get("columns") or []
    names: set[str] = set()
    if isinstance(cols, list):
        for item in cols:
            if isinstance(item, dict):
                name = item.get("name") or item.get("column") or item.get("column_name")
                if name:
                    names.add(str(name).strip())
            elif isinstance(item, str) and item.strip():
                names.add(item.strip())
    elif isinstance(cols, dict):
        names = {str(k).strip() for k in cols if str(k).strip()}
    return names


def _bq_columns(client: Any, project: str, dataset: str, table: str) -> set[str]:
    sql = f"""
    SELECT column_name
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @table_name
    """
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("table_name", "STRING", table)]
    )
    rows = client.query(sql, job_config=job_config).result()
    return {str(r.column_name) for r in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default="", help="Comma-separated subset (default: allowlist ∩ YAML)")
    parser.add_argument("--dataset", default="", help="Override BQ dataset (default BQ_DATASET_GOLD or mart_dev)")
    args = parser.parse_args(argv)
    _load_dotenv()

    project = (os.environ.get("BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    dataset = (args.dataset or os.environ.get("BQ_DATASET_GOLD") or "mart_dev").strip()
    if not project:
        print("ERROR: BQ_PROJECT (or GOOGLE_CLOUD_PROJECT) required", file=sys.stderr)
        return 2

    only = [t.strip() for t in args.tables.split(",") if t.strip()] or None
    candidates = _allowlisted_tables(only)
    # Only tables that have YAML (offline catalog)
    tables = [t for t in candidates if (YAML_DIR / f"{t}.yml").is_file() or (YAML_DIR / f"{t}.yaml").is_file()]
    if not tables:
        print("ERROR: no YAML-backed allowlisted tables to check", file=sys.stderr)
        return 2

    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    drifts: list[str] = []
    checked = 0
    for table in tables:
        ypath = YAML_DIR / f"{table}.yml"
        if not ypath.is_file():
            ypath = YAML_DIR / f"{table}.yaml"
        yaml_cols = _yaml_columns(ypath)
        if not yaml_cols:
            drifts.append(f"{table}: YAML has no parseable columns")
            continue
        try:
            bq_cols = _bq_columns(client, project, dataset, table)
        except Exception as exc:  # pragma: no cover - live BQ
            drifts.append(f"{table}: BQ fetch failed: {exc}")
            continue
        if not bq_cols:
            drifts.append(f"{table}: missing in BQ INFORMATION_SCHEMA")
            continue
        checked += 1
        only_yaml = sorted(yaml_cols - bq_cols)
        only_bq = sorted(bq_cols - yaml_cols)
        if only_yaml or only_bq:
            parts = []
            if only_yaml:
                parts.append(f"yaml_only={only_yaml[:20]}")
            if only_bq:
                parts.append(f"bq_only={only_bq[:20]}")
            drifts.append(f"{table}: " + "; ".join(parts))

    print(f"checked={checked} tables project={project} dataset={dataset}")
    if drifts:
        print("DRIFT DETECTED:")
        for line in drifts:
            print(f"  - {line}")
        return 1
    print("ok: YAML columns match BQ INFORMATION_SCHEMA for allowlisted tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
