#!/usr/bin/env python3
"""Build schema_cards/*.yaml from mart_indicator_classes + mart table YAMLs."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ml-eng"))

from ml.rag.chatbot.mart_indicator_classes import all_class_codes, facts_for_class  # noqa: E402

OUT = ROOT / "ml-eng" / "ml" / "rag" / "schema_cards"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for code in all_class_codes():
        path = OUT / f"{code}.yaml"
        if path.is_file():
            continue
        tables = facts_for_class(code)
        card = {
            "class": code,
            "default_table": tables[0] if tables else "",
            "tables": tables,
            "columns": {},
            "hard_rules": [],
            "exclude_prompt_columns": ["geo_key", "scenario_key", "classification_key", "product_key"],
        }
        path.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
