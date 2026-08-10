"""Patch all staging BQ YAMLs with semantic_relationships and refresh from map."""
from __future__ import annotations

from pathlib import Path

import yaml

from ml.rag.helpers.staging_semantic_relationships import relationships_for

OUT = Path(__file__).resolve().parents[1] / "bq_tables_yaml_files"


def main() -> int:
    count = 0
    for path in sorted(OUT.glob("stg_*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        tid = path.stem
        data["semantic_relationships"] = relationships_for(tid)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        count += 1
    print(f"patched {count} yamls in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
