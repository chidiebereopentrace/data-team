"""
Retrieval@k smoke eval against live Qdrant collections.

Usage:
  PYTHONPATH=ml-eng python -m ml.rag.eval.run_retrieval_eval --corpus news --k 5
  PYTHONPATH=ml-eng python -m ml.rag.eval.run_retrieval_eval --corpus all --k 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from ml.rag.local_env import load_rag_dotenv
from ml.rag.retrievers.vector_retriever import VectorRetriever
from ml.rag.text_processors.chunking_config import PROFILES, CorpusKey

# Ensure the unified env (config/.env + data/local for local dev) is loaded
# before any retriever or Qdrant client is created.
_repo_root = Path(__file__).resolve().parents[3]   # data-team root (contains ml-eng/)
load_rag_dotenv(_repo_root / "ml-eng")

EVAL_DIR = Path(__file__).resolve().parent / "questions"

CORPUS_FILES: dict[CorpusKey, str] = {
    "news": "news.yaml",
    "research": "research.yaml",
}


def _load_questions(corpus: CorpusKey) -> list[dict[str, Any]]:
    path = EVAL_DIR / CORPUS_FILES[corpus]
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data or [])


def _hit_at_k(results: list[dict[str, Any]], expect_doc_kind: str, k: int) -> bool:
    for row in results[:k]:
        meta = row.get("metadata") or {}
        if meta.get("doc_kind") == expect_doc_kind:
            return True
    return False


def eval_corpus(*, corpus: CorpusKey, k: int) -> tuple[int, int]:
    prof = PROFILES[corpus]
    vr = VectorRetriever(collection_name=prof.qdrant_collection)
    mode = prof.qdrant_vector_mode
    questions = _load_questions(corpus)
    if not questions:
        print(f"[{corpus}] no questions in {EVAL_DIR / CORPUS_FILES[corpus]}")
        return 0, 0

    hits = 0
    for item in questions:
        q = str(item.get("query", "")).strip()
        expect = str(item.get("expect_doc_kind", "")).strip()
        if not q or not expect:
            continue
        kwargs: dict[str, Any] = {"top_k": k, "vector_search_mode": mode}
        if expect:
            kwargs["doc_kind"] = expect
        results = vr.retrieve(q, **kwargs)
        ok = _hit_at_k(results, expect, k)
        hits += int(ok)
        status = "HIT" if ok else "MISS"
        print(f"  [{status}] {q[:72]}")

    total = len(questions)
    rate = (hits / total * 100) if total else 0.0
    print(f"[{corpus}] recall@{k}: {hits}/{total} ({rate:.0f}%)")
    return hits, total


def main() -> int:
    p = argparse.ArgumentParser(description="Retrieval@k eval for RAG corpora.")
    p.add_argument("--corpus", choices=["news", "research", "all"], default="all")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    corpora: list[CorpusKey]
    if args.corpus == "all":
        corpora = ["news", "research"]
    else:
        corpora = [args.corpus]  # type: ignore[list-item]

    total_hits = 0
    total_q = 0
    for corpus in corpora:
        h, t = eval_corpus(corpus=corpus, k=int(args.k))
        total_hits += h
        total_q += t

    if total_q:
        print(f"\nOverall recall@{args.k}: {total_hits}/{total_q} ({100 * total_hits / total_q:.0f}%)")
    return 0 if total_hits == total_q or total_q == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
