#!/usr/bin/env python3
"""Merge curated mart semantics + indicator classes into bq_mart_tables_yaml_files/*.yml."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SEMANTICS_YAML = Path(__file__).resolve().parent / "mart_table_semantics.yaml"
MART_YAML_DIR = REPO_ROOT / "ml-eng" / "ml" / "rag" / "bq_mart_tables_yaml_files"
ENTITY_SEED = REPO_ROOT / "data-eng" / "docs" / "mart_entity_dictionary_seed.yaml"

CURATED_KEYS = frozenset(
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


def _load_semantics() -> dict[str, dict[str, Any]]:
    if not SEMANTICS_YAML.is_file():
        return {}
    data = yaml.safe_load(SEMANTICS_YAML.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    tables = data.get("tables") or {}
    return {str(k): v for k, v in tables.items() if isinstance(v, dict)}


def _load_entity_baseline() -> dict[str, dict[str, Any]]:
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
        if not name:
            continue
        domain = str(entry.get("domain") or "").strip()
        grain = str(entry.get("grain") or "").strip()
        purpose = str(entry.get("purpose") or entry.get("analytical_use") or "").strip()
        baseline: dict[str, Any] = {}
        if domain:
            baseline["semantic_role"] = {
                "primary_domain": domain,
                "supports": [domain],
            }
        hints: list[str] = []
        if grain:
            hints.append(f"Grain: {grain}")
        if purpose:
            hints.append(purpose[:200])
        grain_warn = str(entry.get("grain_warning") or "").strip()
        if grain_warn:
            hints.append(grain_warn)
        if hints:
            baseline["filtering_guidance"] = hints
        join_keys = entry.get("join_keys")
        if join_keys:
            baseline["sql_generation_hints"] = [
                f"Join keys: {join_keys}",
                "LEFT JOIN dim_geography on geography_key for country names",
            ]
        out[name] = baseline
    return out


def _merge_into_table_yaml(
    table_name: str,
    payload: dict[str, Any],
    curated: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(payload)
    for key in CURATED_KEYS:
        if key in curated and curated[key]:
            merged[key] = curated[key]
    return merged


def patch_all(*, dry_run: bool = False) -> dict[str, int]:
    curated_tables = _load_semantics()
    baseline = _load_entity_baseline()
    if not MART_YAML_DIR.is_dir():
        print(f"Missing mart YAML dir: {MART_YAML_DIR}", file=sys.stderr)
        return {"patched": 0, "skipped": 0}

    patched = 0
    skipped = 0
    for path in sorted(MART_YAML_DIR.glob("*.yml")):
        table_name = path.stem
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            print(f"skip {table_name}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if not isinstance(payload, dict):
            skipped += 1
            continue
        curated = dict(baseline.get(table_name) or {})
        if table_name in curated_tables:
            for k, v in curated_tables[table_name].items():
                curated[k] = v
        if not curated:
            skipped += 1
            continue
        merged = _merge_into_table_yaml(table_name, payload, curated)
        if dry_run:
            print(f"would patch {table_name}")
        else:
            path.write_text(
                yaml.safe_dump(merged, sort_keys=False, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )
        patched += 1
    return {"patched": patched, "skipped": skipped}


def main() -> int:
    dry = "--dry-run" in sys.argv
    stats = patch_all(dry_run=dry)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
