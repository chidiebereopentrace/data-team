#!/usr/bin/env python3
"""Bootstrap missing mart table YAMLs from entity seed + dbt schema.yml (no BigQuery).

Use when BQ credentials are unavailable. Run full regen when possible:
  python data-eng/data/local/scripts/regenerate_mart_table_yamls.py

Usage:
  python data-eng/data/local/scripts/bootstrap_mart_yaml_skeletons.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

DATA_ENG_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = DATA_ENG_ROOT.parent
ENTITY_SEED = DATA_ENG_ROOT / "docs" / "mart_entity_dictionary_seed.yaml"
SCHEMA_YML = DATA_ENG_ROOT / "dbt" / "models" / "mart_dev" / "schema.yml"
YAML_DIR = REPO_ROOT / "ml-eng" / "ml" / "rag" / "bq_mart_tables_yaml_files"
TABLES_FILE = Path(__file__).resolve().parent / "mart_dev_rag_tables.txt"

sys.path.insert(0, str(REPO_ROOT / "ml-eng"))
from ml.rag.mart_yaml_contract import default_bind_spine, ensure_bind_contract_fields  # noqa: E402


def _load_entity_meta() -> dict[str, dict]:
    if not ENTITY_SEED.is_file():
        return {}
    data = yaml.safe_load(ENTITY_SEED.read_text(encoding="utf-8")) or {}
    out: dict[str, dict] = {}
    for entry in data.get("entities") or []:
        if isinstance(entry, dict):
            name = str(entry.get("table_name") or "").strip()
            if name:
                out[name] = entry
    return out


def _load_schema_columns() -> dict[str, list[dict[str, str]]]:
    if not SCHEMA_YML.is_file():
        return {}
    data = yaml.safe_load(SCHEMA_YML.read_text(encoding="utf-8")) or {}
    out: dict[str, list[dict[str, str]]] = {}
    for model in data.get("models") or []:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name") or "").strip()
        if not name:
            continue
        cols: list[dict[str, str]] = []
        for col in model.get("columns") or []:
            if isinstance(col, dict):
                cname = str(col.get("name") or "").strip()
                if cname:
                    cols.append({"name": cname, "type": "STRING", "description": cname.replace("_", " ")})
        out[name] = cols
    return out


def _listed_tables() -> list[str]:
    names: list[str] = []
    for line in TABLES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split()[0])
    return names


def bootstrap_table(table_id: str, entity: dict, schema_cols: list[dict[str, str]], project: str) -> dict:
    col_names = {c["name"] for c in schema_cols}
    payload: dict = {
        "table_name": f"{project}.mart_dev.{table_id}",
        "description": str(entity.get("purpose") or entity.get("analytical_use") or table_id.replace("_", " ")),
        "source": {"layer": "mart_dev", "lineage": str(entity.get("source_lineage") or "")},
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "columns": schema_cols or [{"name": "placeholder_key", "type": "STRING", "description": "bootstrap placeholder"}],
    }
    if entity.get("grain"):
        payload["grain"] = entity["grain"]
    if entity.get("domain"):
        payload["entity_type"] = entity["domain"]
    spine = default_bind_spine(table_id, col_names)
    if spine:
        payload["bind_spine"] = spine
    return ensure_bind_contract_fields(table_id, payload)


def main() -> int:
    project = "opentrace-prod-5ga4"
    entity_meta = _load_entity_meta()
    schema_cols = _load_schema_columns()
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    created = 0
    for table_id in _listed_tables():
        path = YAML_DIR / f"{table_id}.yml"
        if path.is_file():
            continue
        entity = entity_meta.get(table_id) or {}
        cols = schema_cols.get(table_id) or []
        payload = bootstrap_table(table_id, entity, cols, project)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        created += 1
        print(f"bootstrapped {table_id}")
    print(f"Created {created} skeleton YAML files in {YAML_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
