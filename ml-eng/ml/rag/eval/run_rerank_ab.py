"""Reranker A/B evaluation: retrieval quality BEFORE vs AFTER rerank.

Sprint 1, Week 3 (TASKS.md §2): measures and records the effect of the reranker
on the standard eval question sets, against live Qdrant.

For each question we:
  1. Retrieve a pool of top-N candidates (pre-rerank order = vector score order).
  2. Run the production ``rerank()`` (default cross_encoder mode) to get the
     post-rerank order.
  3. Score both orderings with:
       - recall@k   : is a relevant doc (matching expect_doc_kind) in the top-k?
       - MRR        : reciprocal rank of the first relevant doc (rank quality).
  4. Report BEFORE vs AFTER aggregates so the lift from reranking is explicit.

Relevance label here is the same weak signal the existing smoke eval uses
(``expect_doc_kind`` match). It is a proxy, not human-graded relevance, but it is
consistent across the before/after comparison so the *delta* is meaningful.

Usage:
  python -m ml.rag.eval.run_rerank_ab --corpus all --pool 30 --k 10
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from ml.rag.local_env import load_rag_dotenv
from ml.rag.chatbot.reranker import _reranker_mode, rerank
from ml.rag.retrievers.vector_retriever import VectorRetriever
from ml.rag.text_processors.chunking_config import PROFILES, CorpusKey

_repo_root = Path(__file__).resolve().parents[3]
load_rag_dotenv(_repo_root / "ml-eng")

# Fallback: load config/.env directly (same pattern as validate_namespaces.py)
# in case load_rag_dotenv did not populate QDRANT_URL / QDRANT_API_KEY.
_env_path = Path(__file__).resolve().parents[3] / "config" / ".env"

if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


EVAL_DIR = Path(__file__).resolve().parent / "questions"

CORPUS_FILES: dict[CorpusKey, str] = {
    "news": "news.yaml",
    "research": "research.yaml",
}


def _load_questions(corpus: CorpusKey) -> list[dict[str, Any]]:
    path = EVAL_DIR / CORPUS_FILES[corpus]
    if not path.exists():
        return []
    return list(yaml.safe_load(path.read_text(encoding="utf-8")) or [])


def _is_relevant(row: dict[str, Any], expect_doc_kind: str) -> bool:
    meta = row.get("metadata") or {}
    return bool(expect_doc_kind) and meta.get("doc_kind") == expect_doc_kind


def _recall_at_k(rows: list[dict[str, Any]], expect: str, k: int) -> int:
    return int(any(_is_relevant(r, expect) for r in rows[:k]))


def _reciprocal_rank(rows: list[dict[str, Any]], expect: str) -> float:
    for i, r in enumerate(rows, start=1):
        if _is_relevant(r, expect):
            return 1.0 / i
    return 0.0


def eval_corpus(*, corpus: CorpusKey, pool: int, k: int) -> dict[str, Any]:
    prof = PROFILES[corpus]
    vr = VectorRetriever(collection_name=prof.qdrant_collection)
    mode = prof.qdrant_vector_mode
    questions = _load_questions(corpus)

    n = 0
    before_recall = after_recall = 0
    before_mrr = after_mrr = 0.0

    for item in questions:
        q = str(item.get("query", "")).strip()
        expect = str(item.get("expect_doc_kind", "")).strip()
        if not q or not expect:
            continue

        kwargs: dict[str, Any] = {"top_k": pool, "vector_search_mode": mode, "doc_kind": expect}
        pre = vr.retrieve(q, **kwargs)
        if not pre:
            print(f"  [SKIP] no candidates: {q[:64]}")
            continue

        post = rerank(q, list(pre), top_k=pool)

        n += 1
        b_r = _recall_at_k(pre, expect, k)
        a_r = _recall_at_k(post, expect, k)
        b_m = _reciprocal_rank(pre, expect)
        a_m = _reciprocal_rank(post, expect)
        before_recall += b_r
        after_recall += a_r
        before_mrr += b_m
        after_mrr += a_m

        arrow = "→"
        print(
            f"  {q[:56]:56s} recall@{k}: {b_r}{arrow}{a_r}  "
            f"RR: {b_m:.2f}{arrow}{a_m:.2f}"
        )

    return {
        "corpus": corpus,
        "n": n,
        "before_recall": before_recall,
        "after_recall": after_recall,
        "before_mrr": before_mrr,
        "after_mrr": after_mrr,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["news", "research", "all"], default="all")
    p.add_argument("--pool", type=int, default=30, help="Candidate pool retrieved before reranking.")
    p.add_argument("--k", type=int, default=10, help="Cutoff for recall@k.")
    args = p.parse_args()

    corpora: list[CorpusKey]
    corpora = ["news", "research"] if args.corpus == "all" else [args.corpus]  # type: ignore[list-item]

    print(f"Reranker mode: {_reranker_mode()}  |  pool={args.pool}  k={args.k}\n")

    agg = []
    for corpus in corpora:
        print(f"=== {corpus} ===")
        agg.append(eval_corpus(corpus=corpus, pool=args.pool, k=args.k))
        print()

    tot_n = sum(a["n"] for a in agg)
    print("=" * 64)
    print("SUMMARY (before → after rerank)")
    print("=" * 64)
    for a in agg:
        if a["n"] == 0:
            print(f"  {a['corpus']:16s} (no scored questions)")
            continue
        br = 100.0 * a["before_recall"] / a["n"]
        ar = 100.0 * a["after_recall"] / a["n"]
        bm = a["before_mrr"] / a["n"]
        am = a["after_mrr"] / a["n"]
        print(
            f"  {a['corpus']:16s} n={a['n']:2d}  "
            f"recall@{args.k}: {br:.0f}% → {ar:.0f}%   MRR: {bm:.3f} → {am:.3f}"
        )

    if tot_n:
        tbr = 100.0 * sum(a["before_recall"] for a in agg) / tot_n
        tar = 100.0 * sum(a["after_recall"] for a in agg) / tot_n
        tbm = sum(a["before_mrr"] for a in agg) / tot_n
        tam = sum(a["after_mrr"] for a in agg) / tot_n
        print("-" * 64)
        print(
            f"  {'OVERALL':16s} n={tot_n:2d}  "
            f"recall@{args.k}: {tbr:.0f}% → {tar:.0f}%   MRR: {tbm:.3f} → {tam:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
