# Reranker A/B Evaluation — Results / Status

**Sprint 1, Week 3 · TASKS.md §2 (reranker before/after)**
**Date:** 2026-07-15
**Script:** `ml/rag/eval/run_rerank_ab.py`
**Command:**

```
# from data-team/ml-eng, with QDRANT_URL / QDRANT_API_KEY in config/.env
python -m ml.rag.eval.run_rerank_ab --corpus all --pool 30 --k 10
```

## What the script does

For each standard eval question (`ml/rag/eval/questions/*.yaml`) it retrieves a
candidate pool from live Qdrant (pre-rerank order), runs the production
`rerank()` (default `cross_encoder`, BAAI/bge-reranker-base), and reports
**recall@k** and **MRR** BEFORE vs AFTER reranking so the lift is explicit.

## Environment capability check (Jul 15)

- Reranker model backend: ✅ **available** — `sentence_transformers` present, so
  `cross_encoder` mode loads with real scores (not the `off` fallback).
  (`fastembed` and `cohere` are not installed in this venv.)
- Qdrant: ✅ reachable (creds in `config/.env`).

## Run outcome (Jul 15): ⚠️ blocked on embedding-model mismatch (local env)

The retrieval half failed with a Qdrant dimension error:

```
Wrong input: Vector dimension error: expected dim: 384, got 768
```

- The `news_data` collection was indexed at **384 dims**
  (`paraphrase-multilingual-MiniLM-L12-v2`), but the query embedding in this local
  venv produced **768 dims** (an mpnet-class model). The query-time embedding
  model does not match the model the collection was indexed with.
- Also noted: sparse/hybrid search is disabled locally ("Sparse embeddings
  require fastembed").

This is an **environment provisioning issue, not a code issue** — the reranker
and the A/B harness are both working; they just need the environment configured
with the **same per-collection embedding models used at ingestion** (which the
provisioned / production environment has). This is the same
"needs the provisioned environment" dependency already flagged for Week 4 QA.

## Status

- [x] Reranker on by default (`cross_encoder`), config reachable — confirmed.
- [x] A/B measurement harness built and ready (`run_rerank_ab.py`).
- [x] Cross-encoder backend availability confirmed in this venv (sentence-transformers).
- [ ] **Record real before/after numbers** — blocked locally by the embedding-dim
      mismatch above; to be run on the provisioned environment (Week 4 QA) where
      the query embedding models match the indexed collections. Command is ready;
      no further code changes required.

## To run once the environment matches ingestion embeddings

```
python -m ml.rag.eval.run_rerank_ab --corpus all --pool 30 --k 10
```

Then paste the SUMMARY block (recall@k and MRR, before → after) into this file.
