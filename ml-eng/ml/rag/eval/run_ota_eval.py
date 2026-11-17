"""Replay the Continental OTA fixtures against the current RAG graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_DATASETS = {
    "multilingual": _EVAL_DIR / "questions" / "continental_ota_eval_500.jsonl",
    "complex": _EVAL_DIR / "questions" / "continental_ota_complex_eval_500.jsonl",
}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _evidence_summary(result: dict[str, Any], requested: dict[str, bool]) -> dict[str, Any]:
    from ml.rag.chatbot.generator import is_usable_context_item, is_usable_structured_bq_row

    facts = sum(is_usable_structured_bq_row(row) for row in _rows(result.get("bq_results")))
    documents = sum(
        is_usable_context_item(row) and bool(str(row.get("content") or row.get("text") or "").strip())
        for row in _rows(result.get("vector_results"))
    )
    sql_attempts = _rows(result.get("bq_sql_debug"))
    selection = result.get("corpus_selection") or {}
    corpora = selection.get("results") or {}
    return {
        "mart_facts": {
            "requested": requested["mart_facts"],
            "candidates": facts,
            "skipped_reason": result.get("bq_skipped_reason"),
            "cache_hit": result.get("bq_cache_hit"),
            "execution_ms": result.get("bq_execute_ms"),
            "nl2sql_ms": result.get("bq_nl2sql_ms"),
            "sql_status_counts": dict(Counter(row.get("status") or "unknown" for row in sql_attempts)),
        },
        "documents": {
            "requested": requested["documents"],
            "candidates": documents,
            "selection_reason": selection.get("rationale"),
            "cache_hit": result.get("vector_cache_hit"),
            "retrieval_ms": result.get("vector_ms"),
            "corpus_statuses": {
                corpus: details.get("status") or "unknown"
                for corpus, details in corpora.items()
                if isinstance(details, dict)
            },
        },
    }


def run_case(
    case: dict[str, Any],
    invoke: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run one conversation with its recorded date and actual assistant replies."""
    from ml.rag.chatbot.chat_memory import flat_messages_to_memory

    history: list[dict[str, str]] = []
    session_context: dict[str, Any] = {}
    outcomes = []
    for index, turn in enumerate(case["turns"], 1):
        query = turn["user"]
        expected = turn["expect"]
        summary, recent = flat_messages_to_memory(history)
        started = time.perf_counter()
        error_kind = None
        try:
            result = invoke(
                query,
                reference_date=case["reference_date"],
                conversation_summary=summary,
                recent_turns=recent,
                **session_context,
            )
        except Exception as exc:
            result = {}
            error_kind = type(exc).__name__
        latency = time.perf_counter() - started
        answer = str(result.get("answer") or "")
        evidence = _evidence_summary(result, expected["requested_evidence"])
        run_error = error_kind or ("graph_error" if result.get("error") else None)
        outcomes.append({
            "case_id": case["id"],
            "locale": case["locale"],
            "split": case["split"],
            "reference_date": case["reference_date"],
            "categories": case.get("categories", []),
            "turn": index,
            "latency_s": round(latency, 3),
            "query": query,
            "answer": answer,
            "expected_answer_language": expected.get("answer_language"),
            "reported_answer_language": result.get("answer_lang"),
            "catalogue_indicator_ids": turn["catalogue_scope"]["indicator_ids"],
            "evidence": evidence,
            "missing_requested_evidence": (
                None if run_error else [
                    name for name, leg in evidence.items() if leg["requested"] and not leg["candidates"]
                ]
            ),
            "citations": _rows(result.get("citations")),
            "comparison_gaps": _rows(result.get("comparison_gaps")),
            "fact_requirement_coverage": _rows(result.get("fact_requirement_coverage")),
            "document_requirement_coverage": _rows(result.get("document_requirement_coverage")),
            "source_selection": result.get("source_selection"),
            "usage": result.get("usage") or {},
            "error_kind": run_error,
            "ground_truth_score": None,
        })
        if run_error:
            break
        history.extend((
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ))
        for key, expected_type in (("active_request", dict), ("last_bq_facts", list)):
            if isinstance(result.get(key), expected_type) and result[key]:
                session_context[key] = result[key]
    return outcomes


def _summary(cases: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(row["latency_s"] for row in outcomes)
    return {
        "selected_cases": len(cases),
        "expected_turns": sum(len(case["turns"]) for case in cases),
        "attempted_turns": len(outcomes),
        "errors": sum(bool(row["error_kind"]) for row in outcomes),
        "turns_missing_requested_evidence": sum(bool(row["missing_requested_evidence"]) for row in outcomes),
        "fact_requirement_statuses": dict(Counter(
            entry["status"] for row in outcomes for entry in row.get("fact_requirement_coverage", [])
        )),
        "document_requirement_statuses": dict(Counter(
            entry["status"] for row in outcomes for entry in row.get("document_requirement_coverage", [])
        )),
        "latency_p95_s": latencies[math.ceil(len(latencies) * 0.95) - 1] if latencies else None,
        "gold_scored_turns": 0,
    }


def _load_cases(raw: bytes) -> list[dict[str, Any]]:
    cases = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            for field in ("id", "locale", "split", "kind", "reference_date"):
                if not isinstance(case[field], str) or not case[field]:
                    raise ValueError(f"invalid {field}")
            if case["split"] not in ("development", "held_out"):
                raise ValueError("invalid split")
            date.fromisoformat(case["reference_date"])
            if not isinstance(case["turns"], list) or not case["turns"]:
                raise ValueError("missing turns")
            for turn in case["turns"]:
                if not isinstance(turn["user"], str) or not turn["user"].strip():
                    raise ValueError("empty user prompt")
                ids = turn["catalogue_scope"]["indicator_ids"]
                if not isinstance(ids, list) or not ids or any(not isinstance(id_, str) or not id_ for id_ in ids):
                    raise ValueError("invalid catalogue IDs")
                requested = turn["expect"]["requested_evidence"]
                if any(type(requested[leg]) is not bool for leg in ("mart_facts", "documents")):
                    raise ValueError("invalid evidence flags")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"line {number}: {exc}") from exc
        cases.append(case)
    return cases


def _git_metadata() -> dict[str, Any]:
    def read_git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=_EVAL_DIR, stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    changes = read_git("status", "--porcelain", "--untracked-files=no")
    return {"commit": read_git("rev-parse", "HEAD"), "tracked_changes": bool(changes) if changes is not None else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=_DATASETS)
    parser.add_argument("--split", choices=["development", "held_out", "all"], default="development")
    parser.add_argument("--case", action="append", dest="case_ids", help="select a case by ID (repeatable)")
    parser.add_argument("--limit", type=int, help="max cases (0 means all; default 10 for an unfiltered replay)")
    parser.add_argument("--inventory", action="store_true", help="list selected fixtures without running the graph")
    parser.add_argument("--output", type=Path, help="path for the JSON report")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    cases: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for name in dict.fromkeys(args.dataset or list(_DATASETS)):
        raw = _DATASETS[name].read_bytes()
        fingerprints[name] = hashlib.sha256(raw).hexdigest()
        try:
            cases.extend(_load_cases(raw))
        except ValueError as exc:
            parser.error(f"{_DATASETS[name].name}: {exc}")
    if len({case["id"] for case in cases}) != len(cases):
        parser.error("fixture case IDs must be unique")
    selected = [case for case in cases if args.split == "all" or case["split"] == args.split]
    if args.case_ids:
        by_id = {case["id"]: case for case in selected}
        missing = [case_id for case_id in args.case_ids if case_id not in by_id]
        if missing:
            parser.error(f"case IDs outside the selected fixtures or split: {', '.join(missing)}")
        selected = [by_id[case_id] for case_id in dict.fromkeys(args.case_ids)]
    limit = args.limit if args.limit is not None else (0 if args.inventory or args.case_ids else 10)
    if limit:
        selected = selected[:limit]
    if not selected:
        parser.error("no cases match the requested selection")

    if args.inventory:
        print(json.dumps({
            "cases": len(selected),
            "turns": sum(len(case["turns"]) for case in selected),
            "by_split": Counter(case["split"] for case in selected),
            "by_language": Counter(case["locale"] for case in selected),
            "by_kind": Counter(case["kind"] for case in selected),
            "gold_scored_turns": 0,
        }, indent=2))
        return 0

    from ml.rag.local_env import load_rag_dotenv

    load_rag_dotenv(_EVAL_DIR.parents[2])

    from ml.rag.chatbot.graph import run_rag
    from ml.rag.llm_chat import llm_default_timeout_s
    from ml.rag.llm_model_config import resolved_llm_models

    outcomes: list[dict[str, Any]] = []
    for case in selected:
        print(f"Running {case['id']}", flush=True)
        outcomes.extend(run_case(case, run_rag))
    summary = _summary(selected, outcomes)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.output or (_EVAL_DIR / "reports" / f"ota_baseline_{stamp}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "fixture_sha256": fingerprints,
            "runtime": {
                **_git_metadata(),
                "models": resolved_llm_models(),
                "llm_timeout_s": llm_default_timeout_s(),
            },
            "summary": summary,
            "outcomes": outcomes,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"report: {path}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
