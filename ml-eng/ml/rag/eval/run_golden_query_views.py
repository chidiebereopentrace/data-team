"""Offline golden checks for query views and honesty invariants.

Usage (from ml-eng):
  PYTHONPATH=. python -m ml.rag.eval.run_golden_query_views
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from ml.rag.chatbot.query_gate import early_non_rag_route
from ml.rag.chatbot.query_views import build_query_views, user_query_dropped


def _load_cases() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parent / "questions" / "golden_query_views.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = raw.get("cases") or []
    return [c for c in cases if isinstance(c, dict)]


def _expect_dict(case: dict[str, Any]) -> dict[str, Any]:
    raw = case.get("expect")
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items()}


def _check_query_views(case: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    q = str(case.get("query") or "").strip()
    recent = case.get("recent_turns")
    decomp = case.get("decomposition")
    views = build_query_views(
        q,
        recent_turns=recent if isinstance(recent, list) else None,
        decomposition=decomp if isinstance(decomp, dict) else None,
    )
    expect = _expect_dict(case)
    for token in expect.get("user_contains") or []:
        if str(token).lower() not in views.user_query.lower():
            errs.append(f"user_query missing {token!r}")
    min_tok = expect.get("min_user_tokens")
    if isinstance(min_tok, int) and len(views.user_query.split()) < min_tok:
        errs.append(f"user_query too short ({len(views.user_query.split())} < {min_tok})")
    if "context_applied" in expect and bool(views.context_applied) != bool(expect["context_applied"]):
        errs.append(f"context_applied={views.context_applied} expected {expect['context_applied']}")
    if expect.get("user_query_dropped") is False:
        texts = views.vector_texts()
        if user_query_dropped(views, texts):
            errs.append("user_query_dropped=true")
    return errs


def _check_route(case: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    q = str(case.get("query") or "").strip()
    expect = _expect_dict(case)
    route = early_non_rag_route(q)
    want = expect.get("early_route")
    if want and route != want:
        errs.append(f"route={route!r} expected {want!r}")
    return errs


def _check_static(case: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    expect = _expect_dict(case)
    if expect.get("engines_sql_less"):
        from ml.rag.chatbot.sql_compiler import assemble_sql
        from ml.rag.chatbot.sql_request import SqlRequest

        try:
            sql = assemble_sql(
                SqlRequest(class_code="PROD", table_id="fct_production"),
                {"class": "PROD"},
            )
            if sql and "SELECT" in str(sql).upper():
                errs.append("legacy assemble_sql returned SELECT")
        except Exception:
            pass  # refuse/raise is acceptable for stub
    return errs


def main() -> int:
    failures = 0
    for case in _load_cases():
        cid = str(case.get("id") or "?")
        kind = str(case.get("kind") or "")
        if kind == "query_views":
            errs = _check_query_views(case)
        elif kind == "route":
            errs = _check_route(case)
        elif kind == "static":
            errs = _check_static(case)
        else:
            errs = [f"unknown kind {kind!r}"]
        if errs:
            failures += 1
            print(f"FAIL {cid}: {'; '.join(errs)}")
        else:
            print(f"PASS {cid}")
    if failures:
        print(f"{failures} golden case(s) failed")
        return 1
    print("All golden_query_views cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
