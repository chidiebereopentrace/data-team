"""Price demo smoke: price questions from questions.yaml through the full graph.

Reports, per question, whether the warehouse SQL came from a code template or
NL2SQL, whether a usable warehouse row came back, which vector collections were
cited, and must_mention / must_not_mention violations. Price SQL is also checked
against the mart vocabulary (price_type / price_source literals, unit + currency).

Usage:
  cd ml-eng && python -m ml.rag.eval.run_price_demo --limit 10
  cd ml-eng && python -m ml.rag.eval.run_price_demo --ids e001,e018
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.rag.local_env import load_rag_dotenv

load_rag_dotenv(Path(__file__).resolve().parents[3])

from ml.rag.chatbot.generator import is_usable_structured_bq_row
from ml.rag.chatbot.graph import run_rag
from ml.rag.chatbot.price_vocab import price_sql_vocab_errors

_QUESTIONS = Path(__file__).resolve().parent / "questions" / "questions.yaml"
_REPORTS = Path(__file__).resolve().parent / "reports"
_PRICE_TABLE_RE = re.compile(r"\b(fct_prices|agg_prices_country_month)\b")


def price_questions() -> list[dict[str, Any]]:
    data = yaml.safe_load(_QUESTIONS.read_text(encoding="utf-8")) or []
    out = []
    for item in data:
        expect = item.get("expect") or {}
        if item.get("lang") != "en":
            continue
        if item.get("category") in ("price", "compare") or (
            item.get("category") == "relative_time" and expect.get("measure") == "market_price"
        ):
            out.append(item)
    return out


def _collection(citation: dict[str, Any]) -> str:
    meta = citation.get("metadata") if isinstance(citation.get("metadata"), dict) else {}
    for key in ("corpus", "_context_kind", "collection", "source"):
        value = citation.get(key) or meta.get(key)
        if value:
            return str(value)
    return "unknown"


def run_one(item: dict[str, Any]) -> dict[str, Any]:
    profile = item.get("user_profile") if isinstance(item.get("user_profile"), dict) else {}
    expect = item.get("expect") or {}
    t0 = time.perf_counter()
    error = ""
    try:
        result = run_rag(
            str(item.get("question") or ""),
            user_profile=profile,
            plan_type=profile.get("plan_type"),
            category=profile.get("category"),
        )
    except Exception as exc:
        result = {}
        error = f"{type(exc).__name__}: {exc}"[:300]
    answer = str(result.get("answer") or "")
    debug = [d for d in (result.get("bq_sql_debug") or []) if isinstance(d, dict)]
    rows = [r for r in (result.get("bq_results") or []) if isinstance(r, dict)]
    citations = [c for c in (result.get("citations") or []) if isinstance(c, dict)]
    templates = sorted({str(d.get("template")) for d in debug if d.get("template")})
    sources = sorted({str(d.get("sql_source")) for d in debug if d.get("sql_source")})
    price_sql = [str(d.get("sql") or "") for d in debug if _PRICE_TABLE_RE.search(str(d.get("sql") or ""))]
    vocab_errors = [err for sql in price_sql for err in price_sql_vocab_errors(sql)]
    low = answer.lower()
    missing = [m for m in (expect.get("must_mention") or []) if str(m).lower() not in low]
    leaked = [m for m in (expect.get("must_not_mention") or []) if str(m).lower() in low]
    warehouse_hit = any(is_usable_structured_bq_row(r) for r in rows)
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "route": result.get("route_candidate"),
        "task_mode": result.get("task_mode"),
        "templates": templates,
        "sql_sources": sources,
        "nl2sql": any("nl2sql" in s or "engine" in s for s in sources) and not templates,
        "warehouse_hit": warehouse_hit,
        "corpora": dict(Counter(_collection(c) for c in citations)),
        "missing_mentions": missing,
        "leaked": leaked,
        "vocab_errors": vocab_errors,
        "price_sql": price_sql[:3],
        "latency_s": round(time.perf_counter() - t0, 1),
        "error": error,
        "answer_head": answer[:300],
        "ok": warehouse_hit and not missing and not leaked and not vocab_errors and not error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Price demo smoke over questions.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="comma-separated question ids")
    args = parser.parse_args()
    items = price_questions()
    wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
    if wanted:
        items = [i for i in items if i.get("id") in wanted]
    if args.limit > 0:
        items = items[: args.limit]
    outcomes = []
    for index, item in enumerate(items, 1):
        row = run_one(item)
        outcomes.append(row)
        flag = "PASS" if row["ok"] else "FAIL"
        print(
            f"[{index}/{len(items)}] {flag} {row['id']} bq={int(row['warehouse_hit'])} "
            f"tpl={','.join(row['templates']) or '-'} src={','.join(row['sql_sources']) or '-'} "
            f"corpora={row['corpora']} vocab={len(row['vocab_errors'])} "
            f"miss={row['missing_mentions']} {row['latency_s']}s :: {str(row['question'])[:60]}",
            flush=True,
        )
    passed = sum(r["ok"] for r in outcomes)
    summary = {
        "questions": len(outcomes),
        "passed": passed,
        "warehouse_hit": sum(r["warehouse_hit"] for r in outcomes),
        "template_sql": sum(bool(r["templates"]) for r in outcomes),
        "nl2sql_only": sum(r["nl2sql"] for r in outcomes),
        "vocab_violations": sum(bool(r["vocab_errors"]) for r in outcomes),
    }
    _REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = _REPORTS / f"price_demo_{stamp}.json"
    path.write_text(json.dumps({"summary": summary, "outcomes": outcomes}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report: {path}")
    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
