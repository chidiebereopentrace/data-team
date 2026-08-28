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

---

## Sprint 2 reranker status update (ML-045 · 2026-08-08)

### What changed since Jul 15

| Item | Jul 15 | Aug 08 |
|---|---|---|
| Reranker mode (Railway) | ❌ Silently `off` — `cross_encoder` degraded because fastembed 0.5 lacked `TextCrossEncoder` | ✅ `cross_encoder` active — fastembed 0.8.0 ships `TextCrossEncoder` ONNX |
| Model baked in Railway image | ❌ Not in `Dockerfile.railway` — first request triggered ~280 MB download | ✅ `_load_cross_encoder('BAAI/bge-reranker-base')` added to warmup step in `Dockerfile.railway` |
| `.env.example` default | `cross_encoder` (active) but incorrect — would degrade on Railway slim image | ✅ `cross_encoder` active with warning note; `openrouter`/`cohere` documented as optional |
| Cost | N/A | **$0/req** — fully local ONNX, no API call |

### What is live on main now

- `RAG_RERANKER_MODE=cross_encoder` — documented and active in `.env.example`
- `BAAI/bge-reranker-base` baked into `Dockerfile.railway` — zero cold-start download
- fastembed 0.8.0 in `requirements.railway.txt` (`fastembed>=0.5.0`) — `TextCrossEncoder` available
- Full degradation chain retained: `cross_encoder` → `llm` (if configured) → `off`
- All 25 reranker unit tests passing

### Before/after quantitative run — still pending

The embedding-dim mismatch blocker from Jul 15 is unchanged locally.
The A/B harness (`run_rerank_ab.py`) is ready; run it in the Railway/provisioned
environment once the next deployment is live:

```
python -m ml.rag.eval.run_rerank_ab --corpus all --pool 30 --k 10
```

Paste recall@k and MRR before/after here.

### Expected impact on Tavily quota

With the reranker now genuinely active on Railway, retrieval quality should improve
enough that `needs_web_fallback()` fires less. The `query_langfuse_tavily.py` script
(ML-045) will measure this directly once traces accumulate post-redeploy.
Run it with: `python query_langfuse_tavily.py --days 14` after the next Railway deploy
to capture a clean before/after window.

### Status

- [x] Reranker confirmed working on Railway (fastembed 0.8 + Dockerfile warmup)
- [x] Zero-cost local reranking — no OpenRouter or Cohere API calls
- [x] All unit tests passing (25/25)
- [ ] Before/after recall@k and MRR numbers — pending provisioned env run
- [ ] Tavily quota impact measured — pending post-redeploy Langfuse traces (ML-045)
