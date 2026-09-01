"""Smoke eval: supervisor class routing vs bq_descriptions.yaml expectations.

Usage:
  cd ml-eng && python -m ml.rag.eval.run_bq_routing_eval --verbose
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ml.rag.chatbot.agri_measure_ontology import resolve_measures
from ml.rag.chatbot.class_supervisor import compile_supervisor_plan
from ml.rag.chatbot.facet_enrich import enrich_decomposition_facets
from ml.rag.chatbot.mart_indicator_classes import facts_for_class
from ml.rag.chatbot.query_normalize import normalize_query_text

_YAML = Path(__file__).resolve().parent / "questions" / "bq_descriptions.yaml"


def _load_cases() -> list[dict]:
    if not _YAML.is_file():
        return []
    raw = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else []


def _class_tables(class_code: str) -> set[str]:
    return {t.split(".")[-1].lower() for t in facts_for_class(class_code)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BQ routing smoke eval from bq_descriptions.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    failed = 0
    for case in _load_cases():
        q = normalize_query_text(str(case.get("query") or ""))
        expect = [str(t).strip().lower() for t in (case.get("expect_tables") or []) if str(t).strip()]
        dec = enrich_decomposition_facets(q, {"geography": [], "entities": [], "domains": []})
        hits = resolve_measures(q, dec)
        mh = hits[0] if hits else None
        sp = compile_supervisor_plan(q, decomposition=dec, measure_hit=mh)
        tables = _class_tables(sp.classes[0]) if sp.classes else set()
        ok = any(t in tables for t in expect) if expect and sp.classes else bool(sp.classes)
        if args.verbose or not ok:
            print(f"{'PASS' if ok else 'FAIL'}: {q[:60]} -> {sp.classes} tables={sorted(tables)[:4]} expect={expect}")
        if not ok:
            failed += 1
    print(f"Done: {len(_load_cases()) - failed}/{len(_load_cases())} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
