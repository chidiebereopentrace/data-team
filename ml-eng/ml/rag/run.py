"""
CLI entrypoint for the RAG pipeline. Usage:
  PYTHONPATH=. python -m ml.rag.run "your question here"
  PYTHONPATH=. python -m ml.rag.run

Optional: prints generated BigQuery SQL on **stderr** when ``RAG_CLI_SHOW_SQL=1`` (off by default;
advisory answers never include SQL).

Set ``RAG_CLI_SHOW_RETRIEVAL=0`` to hide a one-line retrieval summary on stderr (counts of
BQ rows, catalog chunks, news, academic, merged). When merged is 0, the model sees no context —
fix env (``BQ_PROJECT``, ``GOOGLE_APPLICATION_CREDENTIALS``, ``QDRANT_URL``, ``QDRANT_API_KEY``, and collection env vars).
BigQuery failures are explained on stderr when ``RAG_BQ_RETRIEVER_DEBUG`` is on (default).

Set ``RAG_CLI_PIPELINE_TRACE=1`` for an extra stderr line: decomposition summary, BQ hint
table names from the matcher, and ``rag_sql_source`` for the first BQ row.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ml.rag.local_env import load_rag_dotenv

load_rag_dotenv(Path(__file__).resolve().parents[2])


def _dedupe_bq_sql(bq_results: list[Any] | None) -> list[str]:
    """Unique SQL strings from BQ retriever rows (metadata.sql on each item)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in bq_results or []:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata")
        if not isinstance(meta, dict):
            continue
        sql = meta.get("sql")
        if not isinstance(sql, str):
            continue
        s = sql.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _should_print_bq_sql() -> bool:
    raw = os.environ.get("RAG_CLI_SHOW_SQL", "0").strip().lower()
    return raw in ("1", "true", "on", "yes")


def _print_bq_sql_stderr(result: dict[str, Any]) -> None:
    """Print generated SQL on stderr so it appears before the answer and is easy to spot."""
    if not _should_print_bq_sql():
        return
    bq = result.get("bq_results") or []
    sqls = _dedupe_bq_sql(bq)
    if not sqls:
        return
    src = ""
    if bq and isinstance(bq[0], dict):
        src = str(bq[0].get("rag_sql_source") or "").strip()
    print("\n--- BigQuery SQL (generated) ---", file=sys.stderr)
    if src:
        print(
            f"(sql_source={src} — nl2sql=LLM; fallback_*=heuristic when enabled)\n",
            file=sys.stderr,
        )
    for i, s in enumerate(sqls, 1):
        print(f"\n[{i}]\n{s}", file=sys.stderr)
    print(file=sys.stderr)


def _should_print_retrieval_summary() -> bool:
    raw = os.environ.get("RAG_CLI_SHOW_RETRIEVAL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "n")


def _should_print_pipeline_trace() -> bool:
    raw = os.environ.get("RAG_CLI_PIPELINE_TRACE", "").strip().lower()
    return raw in ("1", "true", "yes", "on", "y")


def _print_pipeline_trace(result: dict[str, Any]) -> None:
    """Second line: how decompose + BQ hints + sql_source lined up for this run."""
    parts: list[str] = ["[rag] pipeline:"]
    dec = result.get("decomposition")
    if isinstance(dec, dict):
        geo = dec.get("geography") or []
        g0 = geo[0] if isinstance(geo, list) and geo else ""
        ts = (dec.get("time_start") or "")[:10]
        te = (dec.get("time_end") or "")[:10]
        parts.append(f"decompose geo={g0!r} time={ts!r}..{te!r}")
    else:
        parts.append("decompose=(none)")
    cands = result.get("bq_table_candidates") or []
    names: list[str] = []
    for c in cands[:12]:
        if not isinstance(c, dict):
            continue
        md = c.get("metadata")
        if isinstance(md, dict):
            tn = str(md.get("table_name") or "").strip()
            if tn:
                names.append(tn)
    if names:
        parts.append(f"bq_hint_tables=[{', '.join(names)}]")
    else:
        parts.append("bq_hint_tables=[] (vector bq_table_description empty or no table_name)")
    bq = result.get("bq_results") or []
    if bq and isinstance(bq[0], dict):
        src = str(bq[0].get("rag_sql_source") or "").strip()
        if src:
            parts.append(f"first_bq_sql_source={src!r}")
    print(" ".join(parts), file=sys.stderr)


def _print_retrieval_summary(result: dict[str, Any]) -> None:
    """Explain empty answers: generator only sees data when merged_context is non-empty."""
    n_bq = len(result.get("bq_results") or [])
    n_cat = len(result.get("bq_table_candidates") or [])
    n_news = len(result.get("vector_news_results") or [])
    n_acad = len(result.get("vector_academic_results") or [])
    n_merge = len(result.get("merged_context") or [])
    line = (
        f"[rag] retrieval: bq_rows={n_bq} bq_catalog={n_cat} news={n_news} academic={n_acad} "
        f"merged={n_merge}"
    )
    print(line, file=sys.stderr)
    if n_merge == 0:
        print(
            "[rag] hint: no chunks reached the generator. Check BQ_PROJECT, "
            "GOOGLE_APPLICATION_CREDENTIALS, network to BigQuery, and Qdrant "
            "(QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_* for news/research/data descriptions). "
            "If BigQuery failed, stderr may include a line from bq_retrieve explaining why.",
            file=sys.stderr,
        )


def main() -> int:
    from ml.rag.graph import run_rag

    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else (
        "What bronze tables can we query for yields and food security?"
    )
    try:
        result = run_rag(query)
        if _should_print_retrieval_summary():
            _print_retrieval_summary(result)
            if _should_print_pipeline_trace():
                _print_pipeline_trace(result)
        _print_bq_sql_stderr(result)
        answer = result.get("answer", "")
        print(answer)
        if result.get("error"):
            print("Error:", result["error"], file=sys.stderr)
        return 0 if not result.get("error") else 1
    except Exception as e:
        print(f"RAG failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
