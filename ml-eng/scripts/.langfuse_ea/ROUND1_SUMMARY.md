# Ask ADZA Langfuse error analysis — round 1 (2026-07-21)

## Project
- Name: **askADZA**
- Project ID: `cmrieiomo1e93ad0dvoiawwy5`
- Host: https://cloud.langfuse.com

## Setup status
- Langfuse CLI (`npx langfuse-cli`): OK
- Cursor skill: `.cursor/skills/langfuse/` present
- Trace inventory at start: **2** `rag.query` traces (both from unit tests, not production)

## Seed attempt
- Script: `ml-eng/scripts/seed_langfuse_error_analysis.py`
- Local seed **failed**: LM Studio at `RAG_LLM_BASE_URL` returned HTTP 400; embeddings path also missing `fastembed` initially (now installed in the local env).
- No `RAG_LLM_API_KEY` / OpenRouter key in local `.env` to fall back to.
- **Action for you:** run real traffic on Railway (with Langfuse keys) or fix local LLM, then re-run the seed script / more `/query` calls until ~50–100 traces exist. Re-open analysis with a fuller sample.

## Annotation targets
Content lived on root `rag.query` **SPAN** (ERROR level). No GENERATION children.

| Trace | Observation | Notes |
|-------|-------------|-------|
| `df5ece83697d4005278d867fe7a553f6` | `f97298d91f4c522b` | query=`hi`, AssertionError on `RagTraceHandle.span` |
| `106df1d0594fa308062f3101e7320488` | `6e6bd66b2c26009b` | same ERROR pattern |

## Queues
1. **Open coding:** [2026-07-21 Open Coding - Ask ADZA RAG](https://cloud.langfuse.com/project/cmrieiomo1e93ad0dvoiawwy5/annotation-queues/cmrtvez0e0nyead0ifb2afg23)
2. **Taxonomy label queue:** **blocked** — Hobby plan allows only one annotation queue (`Maximum number of annotation queues reached`). Boolean score configs for the taxonomy were still created; label via Scores UI or upgrade plan for a second queue.

Score configs created: `open_coding` (TEXT), `pass_fail_assessment` (Pass/Fail), plus BOOLEAN taxonomy below.

## Open coding (agent first pass on n=2)
Both marked **Fail**. Observations describe unit-test / instrumentation noise, not user-facing RAG quality.

## Failure taxonomy (approved for this round)
| Category | Definition |
|----------|------------|
| `test_instrumentation_noise` | Unit tests / harness leave ERROR `rag.query` spans without a real answer |
| `retrieval_miss` | Missing/wrong evidence for the question |
| `nl2sql_bad` | Bad SQL or BQ path when structured data needed |
| `unsupported_claim` | Answer invents facts not in context |
| `citation_mismatch` | Citations don't match evidence used |
| `wrong_route` | meta/product/full_rag routing wrong |
| `plan_tier_ignored` | Violates plan_type depth/geo/compare rules |
| `web_fallback_noise` | Web fallback hurts answer quality |

## Rates (n=2 — illustrative only)
```
test_instrumentation_noise   100%  ████████████
(all other categories)         0%
pass_fail Fail rate          100%
```

## Decisions (Phase 5)
| Category | Rate | Decision | Next step |
|----------|------|----------|-----------|
| `test_instrumentation_noise` | 100% | **Code fix** | Fix `test_observability.py` expectations around `RagTraceHandle.span` / avoid leaving ERROR root spans; filter `error_analysis_seed` vs test sessions in sampling |
| Remaining RAG categories | n/a | **Monitor until data** | After Railway/real traces exist, re-run open coding on 30–50 real turns, then re-label |

## Your next steps
1. Confirm Langfuse keys on **Railway** API service; run diverse `/query` traffic (or `python scripts/seed_langfuse_error_analysis.py` once LLM works).
2. Open the Label queue and continue annotating new observations (add them via UI or ask the agent to enqueue).
3. When ≥30 real Fail notes exist, re-cluster — taxonomy may split/merge.
4. Optional: fix flaky observability unit tests that create ERROR traces.
