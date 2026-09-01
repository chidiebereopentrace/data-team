#!/usr/bin/env python3
"""Build complete value index JSON from mart table YAMLs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ml-eng"))

from ml.rag.chatbot.bq_table_schema_yaml import (  # noqa: E402
    list_mart_table_index,
    load_mart_table_schema,
    value_samples_for_mart_tables,
)

OUT = ROOT / "ml-eng" / "ml" / "rag" / "value_index" / "mart_value_index.json"
_SAMPLE_SUFFIX = "_value_samples"


def main() -> None:
    enums: dict[str, list[str]] = {}
    fact_scoped: dict[str, list[str]] = {}
    for row in list_mart_table_index():
        tid = str(row.get("table_id") or "").strip()
        if not tid:
            continue
        schema = load_mart_table_schema(tid) or {}
        samples_map = value_samples_for_mart_tables({tid}).get(tid.lower()) or {}
        for key, vals in samples_map.items():
            if not vals:
                continue
            enums[f"{tid.lower()}.{key}"] = list(dict.fromkeys(str(v) for v in vals))
        for key, raw in schema.items():
            if not isinstance(key, str) or not key.endswith(_SAMPLE_SUFFIX):
                continue
            col = key[: -len(_SAMPLE_SUFFIX)]
            if isinstance(raw, list) and raw:
                k = f"{tid.lower()}.{col}"
                merged = list(dict.fromkeys(list(enums.get(k, [])) + [str(v) for v in raw]))
                enums[k] = merged
        wheat = [v for v in enums.get(f"{tid.lower()}.product_name", []) if "wheat" in v.lower()]
        if wheat and tid.lower() == "fct_food_balance":
            fact_scoped[f"{tid.lower()}.product_name"] = wheat

    payload = {"enums": enums, "fact_scoped": fact_scoped}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(enums)} enum keys)")


if __name__ == "__main__":
    main()
