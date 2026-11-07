#!/usr/bin/env python3
"""Sync mart_dev_rag_tables.txt (the RAG table allowlist) from dbt mart_dev SQL models.

Usage (from repo root):
  python data-eng/data/local/scripts/sync_mart_dev_tables_list.py
  python data-eng/data/local/scripts/sync_mart_dev_tables_list.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_ENG_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = DATA_ENG_ROOT.parent
MART_DEV_DIR = DATA_ENG_ROOT / "dbt" / "models" / "mart_dev"
OUT_FILE = Path(__file__).resolve().parent / "mart_dev_rag_tables.txt"

PREFIXES = ("fct_", "agg_", "dim_", "bridge_")
SKIP_PREFIXES = ("gold_",)


def rag_relevant_tables() -> list[str]:
    names: set[str] = set()
    for path in sorted(MART_DEV_DIR.rglob("*.sql")):
        stem = path.stem
        if stem.startswith(SKIP_PREFIXES):
            continue
        if stem.startswith(PREFIXES):
            names.add(stem)
    return sorted(names)


def _section_label(table: str) -> str:
    if table.startswith("dim_"):
        return "DIMENSIONS"
    if table.startswith("bridge_"):
        return "BRIDGE TABLES"
    if table.startswith("fct_"):
        return "FACT TABLES"
    if table.startswith("agg_"):
        return "AGGREGATES"
    return "TABLES"


def format_table_list(tables: list[str]) -> str:
    lines = [
        "# Auto-generated from dbt/models/mart_dev by sync_mart_dev_tables_list.py",
        "# All rag-relevant tables in the mart_dev (Gold) layer.",
        "",
    ]
    current = ""
    for table in tables:
        label = _section_label(table)
        if label != current:
            lines.append(f"# ── {label} ──")
            current = label
        lines.append(table)
    lines.append("")
    return "\n".join(lines)


def read_listed_tables(path: Path) -> list[str]:
    if not path.is_file():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Sync {OUT_FILE.name} from dbt models")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if allowlist differs from dbt (no write)",
    )
    args = parser.parse_args()

    expected = rag_relevant_tables()
    if args.check:
        listed = read_listed_tables(OUT_FILE)
        if listed != expected:
            missing = sorted(set(expected) - set(listed))
            extra = sorted(set(listed) - set(expected))
            print(f"{OUT_FILE.name} out of sync with dbt ({len(listed)} listed, {len(expected)} expected)")
            if missing:
                print(f"  missing from list: {missing}")
            if extra:
                print(f"  extra in list: {extra}")
            return 1
        print(f"{OUT_FILE.name} in sync ({len(expected)} tables)")
        return 0

    OUT_FILE.write_text(format_table_list(expected), encoding="utf-8")
    print(f"Wrote {len(expected)} tables to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
