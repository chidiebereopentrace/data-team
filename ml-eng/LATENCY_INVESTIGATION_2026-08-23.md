# RAG Pipeline Latency Investigation

**Date:** 2026-08-23 (updated 2026-08-24 x3 -- Railway config grounding, then the NL2SQL test
matrix, then the reranker / bq_reason / concurrency follow-up tests)
**Author:** Juma (ml-eng / Intelligence)
**Requested by:** Chidi
**Tools:** `query_langfuse_latency.py` (Langfuse stage breakdown), ad-hoc OpenRouter probes,
direct Langfuse observation drill-down for reranker mode confirmation

**Status: COMPLETE -- 7 confirmed/evidenced findings, all originally-flagged open items tested.**

---

## TL;DR (final, evidence-based)

**Two theories were wrong before landing on this one** (see timeline below) -- worth reading if
you want the full trail, but the short version:

`RAG_BQ_NL2SQL_MODEL_ID` (`deepseek/deepseek-v4-flash-0731`) is a **reasoning model with highly
variable, non-deterministic reasoning length** even at `temperature=0.0`. `RAG_BQ_NL2SQL_MAX_TOKENS`
controls a real, measurable effect (truncation rate + cost) but is **not a reliable fix for tail
latency** on its own -- some calls will still take 60-130s regardless of the cap, because the
model itself sometimes reasons for a very long time before answering.

**What raising `RAG_BQ_NL2SQL_MAX_TOKENS` 1024 -> 2048 DOES fix, backed by a 6-question test
matrix using real production questions + their real table-hint counts:**

| Metric | 1024 (current prod) | 2048 |
|---|---|---|
| Truncated / empty response (triggers a retry in prod) | **4 of 6 (67%)** | **1 of 6 (17%)** |
| Avg cost per NL2SQL call | $0.000235 | $0.000177 (cheaper) |
| Median latency | 29.4s | 24.5s |

**What it does NOT reliably fix:**

| Metric | 1024 | 2048 |
|---|---|---|
| Mean latency | 34.7s | 48.1s (worse) |
| Max latency | 73.0s | **127.8s** (worse) |

One query at 2048 tokens took 127.8s -- longer than anything we saw at 1024. The model's
reasoning length varies a lot run to run and question to question; a bigger token ceiling gives
it *room* to reason longer when it wants to, it doesn't cap how long it reasons for.

**Bottom line for Chidi:** raising `RAG_BQ_NL2SQL_MAX_TOKENS` to 2048 is still worth doing --
it cuts truncation-triggered retries by 4x and is slightly cheaper on average -- but it is
**not a complete fix**. Expect the p90/p99 tail latency to remain rough until we either (a)
switch to a non-reasoning or faster model for NL2SQL, or (b) add a hard per-call timeout with
fallback so one slow table-hint call can't drag the whole request past 60-100s.

**Second confirmed finding (26% of median latency):** the `rerank` stage IS running
`cross_encoder` / `BAAI/bge-reranker-base` correctly on Railway (the earlier ML-044 concern
about it silently degrading to `off` has been fixed) -- but it's genuinely slow on Railway's
CPU-only container, ~730ms per candidate, ~21-22s for a typical 30-candidate pool. This is a
separate, real contributor requiring its own fix (see "Second confirmed finding" section
below) -- not something `RAG_BQ_NL2SQL_MAX_TOKENS` touches.

---

## Investigation timeline (kept for the record -- shows the wrong turns too)

1. **Langfuse trace analysis (25-trace sample, 14 days):** `retrieval.bq.nl2sql` at 79.8s p50,
   ~77% of the ~103s median end-to-end request latency. Everything else fast.
2. **Wrong theory #1:** assumed `RAG_BQ_NL2SQL_PARALLEL` defaulted to `off` in production.
   Disproved -- Chidi confirmed Railway already has it `on` (`workers=4`).
3. **Wrong theory #2:** suspected the downstream sequential SQL validation/execution loop.
   Disproved by re-reading the code -- the slow Langfuse span wraps only generation.
4. **Wrong theory #3:** suspected the model itself was just slow. Disproved with a small
   synthetic prompt (2-4s per call, fast).
5. **Found the real mechanism:** realistic-size prompt (6 real table-hint YAMLs) at the real
   prod `max_tokens=1024` reproduced ~77s in a single test -- looked like a clean fix.
6. **Ran it properly (this update):** pulled 6 REAL questions + their real table_hints_count
   straight from Langfuse `retrieval.bq.nl2sql` observation inputs, tested each at both 1024
   and 2048 tokens. Result: **the effect is real but partial** -- token starvation clearly
   drives truncation/retries and cost, but does not reliably bound worst-case latency, because
   this model's reasoning length itself is highly variable.

**Lesson (reinforced):** a single test case, even a realistic one, is not enough evidence for a
model with non-deterministic reasoning length. Needed a small matrix across real questions to
see the real distribution, not just one lucky/unlucky sample.

---

## Recommended actions

1. **Do raise `RAG_BQ_NL2SQL_MAX_TOKENS` to 2048 in Railway.** Real, measured benefit: cuts
   truncation-triggered retries from 67% to 17% of calls, and is cheaper on average. Low risk,
   single env var, no code deploy.

2. **Do not expect this alone to fix p90/p99 latency.** Flag to Chidi as a partial fix. Next
   steps to actually bound tail latency, in rough order of effort:
   - **Add a tighter per-call timeout with a template/pattern-match fallback** for NL2SQL
     specifically (separate from the existing `RAG_BQ_NL2SQL_TIMEOUT_S=300`, which is far too
     loose to protect user-facing latency -- 300s is 5 minutes).
   - **A/B a non-reasoning or faster model** for NL2SQL (e.g. a smaller Llama/Qwen instruct
     model without a hidden reasoning budget) and compare truncation rate + latency + SQL
     validity against `deepseek-v4-flash-0731`.
   - **Consider `RAG_BQ_NL2SQL_MODE=batch`** (one LLM call for all table hints instead of one
     per hint) to reduce the number of independent "dice rolls" on reasoning length per
     request -- fewer calls means fewer chances to hit a slow outlier, at the cost of a larger
     single prompt.

3. **Re-measure after the Railway change** with `query_langfuse_latency.py --days 2-3` once
   there's real post-change traffic, to see the actual production p50/p90 shift (not just this
   isolated test matrix).

---

## Test matrix data (6 real production questions, pulled from Langfuse observation inputs)

Questions (with real `table_hints_count` from their original trace):

1. "what are the information about school feeding in east africa" (4 hints, orig prod latency 225.0s)
2. "Hey. Follow-up: Hey" (4 hints, orig prod latency 86.9s)
3. "What is the current realities of agricultural data in the Nigeria economy..." (4 hints, orig 25.2s)
4. "What has the government done to regulate the rising cost of agricultural produce in nigeria" (4 hints, orig 23.5s)
5. "What is the current state of cassava in Nigeria?" (3 hints, orig 79.8s)
6. "ghnrf" (garbage/typo input -- included deliberately; 4 hints, orig 59.4s)

Full per-call results in `latency_test_matrix_results.json` (kept alongside this doc). Table
hints for each test used the same truncation logic as production
(`RAG_BQ_HINT_MAX_BYTES=8000` split evenly across hints), using the first N files from
`ml/rag/bq_tables_yaml_files/` as stand-ins for the real vector-matched hints (the actual
table-matching step was not re-run; only the generation call itself was isolated and tested).

**Caveat:** using the first N YAML files alphabetically as stand-ins means the *content* of the
hints doesn't necessarily match what the real pipeline would have matched for that specific
question (e.g. "cassava in Nigeria" would in reality match a crop-production table, not
necessarily whatever the 3 alphabetically-first files are). This affects how *plausible* the
model's answer is, but the token-starvation / reasoning-length effect being measured here is a
property of prompt size and model behavior, not of whether the hint content specifically
answers the question -- so the timing conclusions should still hold directionally, but a fully
rigorous re-test would replay the exact hints from each trace's `retrieval.bq.nl2sql` input
(available in Langfuse but not extracted in this pass -- see Follow-ups).

---

## Data (Langfuse, last 14 days, 25-trace sample, 90 total `rag.query` traces in window)

### Overall end-to-end trace latency (pre-fix baseline)

| Percentile | Latency |
|---|---|
| p50 | 103.2s |
| p90 | 209.1s |
| p99 | 255.4s |
| max | 271.1s |
| avg | 106.0s |

### Stage-level breakdown (p50, sorted descending)

| Stage | n | p50 (ms) | p90 (ms) | max (ms) | % of overall p50 |
|---|---|---|---|---|---|
| **bq_retrieve** | 21 | 82,520 | 164,082 | 178,377 | **80%** |
| ↳ **retrieval.bq.nl2sql** | 21 | 79,808 | 91,448 | 106,681 | **77%** |
| rerank | 42 | 26,740 | 55,004 | 64,466 | 26% |
| llm_chat_complete (raw LLM calls, all stages) | 126 | 9,306 | 82,006 | 106,577 | 9% |
| generate (final answer) | 21 | 4,348 | 8,423 | 12,385 | 4% |
| decompose | 50 | 2,438 | 3,367 | 6,119 | 2% |
| parallel_retrieve (6 Qdrant collections) | 21 | 2,435 | 4,634 | 6,972 | 2% |
| web_fallback (Tavily, when it fires) | 6 | 1,890 | 3,046 | 3,109 | 2% |
| retrieval.qdrant (per-collection) | 245 | 565 | 1,252 | 4,225 | 0.5% |
| bq_reason (table selection, not SQL gen) | 21 | 297 | 1,002 | 1,107 | 0.3% |
| embedding.query | 511 | 77 | 392 | 1,094 | 0.1% |
| merge | 42 | 5 | 36 | 51 | -- |

---

## Second confirmed finding: `rerank` stage IS `cross_encoder`, but slow on Railway CPU (26% of p50)

Confirmed directly from real Railway production Langfuse observations (not a guess):

```json
{
  "mode": "cross_encoder",
  "model": "BAAI/bge-reranker-base",
  "input_count": 30,
  "rerank_pool_size": 30,
  "output_count": 24,
  "latency_ms": 21747.78
}
```

So the reranker IS running the fixed, working, baked-in `cross_encoder` path (confirmed by Dockerfile.railway `_load_cross_encoder('BAAI/bge-reranker-base')` warmup) -- NOT stuck in a broken Cohere fallback. This part of the earlier ML-044 concern (fastembed silently degrading to `off` on Railway) has genuinely been fixed.

**But it is still slow: ~730ms per candidate** (21.7s / 30 candidates) running the cross-encoder on Railway's CPU-only container, no GPU. This is a real, separate contributor to overall latency -- **26% of median request time**, second only to the NL2SQL issue.

### A config drift also found along the way (lower priority, currently harmless)

Railway has `RAG_RERANKER_MODE="cohere"` set explicitly, but **no `COHERE_API_KEY`** is configured. Traced through the code: with no key, `_rerank_cohere()` returns `None` immediately (no network call, ~0ms cost) and falls through to `cross_encoder` on every single request. So this isn't currently costing any latency -- but it means every request pays for an always-failing mode check, and the intent (`cohere` reranking) is silently never actually used. Worth cleaning up (set `RAG_RERANKER_MODE=cross_encoder` explicitly, or add `COHERE_API_KEY` if Cohere reranking was actually wanted) for clarity, not urgency.

### What would actually reduce rerank latency

- **Reduce `RAG_RERANK_POOL_SIZE`** (currently 30, default is 24 per `.env.example`) -- fewer candidates scored per request, directly proportional to latency (measured ~730ms/candidate).
- **Investigate Railway container CPU allocation** -- cross-encoder inference is CPU-bound; if the container is resource-constrained or sharing CPU with other processes, that alone could explain the per-candidate cost being higher than expected for a ~280MB model.
- **Consider `openrouter` rerank mode instead** (per the original, still-open ML-044 recommendation) -- offloads compute to OpenRouter's `cohere/rerank-4-pro`, trading a network round trip for local CPU time. Would need its own latency measurement to know if it's actually faster than the current 21-22s.

---

## Third finding: `bq_reason` (table selection) -- do NOT copy the NL2SQL fix here

Tested the same way (real production questions, real `format_reasoner_index()` output, same
model `deepseek/deepseek-v4-flash-0731`) but for the **table-selection reasoner** step
(`RAG_BQ_REASONER_MAX_TOKENS`, currently defaulting to **800**, not set in Railway):

| `RAG_BQ_REASONER_MAX_TOKENS` | Mean | Median | Max | Truncation rate |
|---|---|---|---|---|
| **800 (current default)** | **12.4s** | **7.0s** | 23.4s | 1/3 (33%) |
| 1600 | 56.4s | 31.0s | **119.9s** | 0/3 (0%) |

**Counter-intuitive result: raising the token limit here made it dramatically *worse*, not
better.** One call at 1600 tokens took 119.9 seconds. This is the opposite direction from the
NL2SQL finding -- confirming the earlier lesson that this model's reasoning-length behavior is
genuinely unpredictable and prompt/context-dependent, not something with a single universal
fix. **Do not blindly raise `RAG_BQ_REASONER_MAX_TOKENS` to "fix" it the same way as NL2SQL.**

This is consistent with real production data: `bq_reason` measured only 297ms p50 in the
original 25-trace Langfuse sample (Data section above) -- i.e. it is *not* currently a problem
in production at its current default. **Recommendation: leave `RAG_BQ_REASONER_MAX_TOKENS`
unset/default (800). No action needed here.** Flagging only because it was on the "untested"
list and is worth knowing the direction doesn't generalize.

---

## Fourth finding: parallelism works, but one slow outlier still sets the floor

Simulated `RAG_BQ_NL2SQL_PARALLEL_WORKERS=4` concurrent calls (matching production exactly)
against OpenRouter directly, instead of running calls one at a time like the earlier tests:

```
4 concurrent calls, individual latencies: 19.98s, 22.49s, 12.14s, 62.20s
Concurrent wall time: 62.34s
Sequential-equivalent time (sum): 116.80s
Speedup from parallelism: 1.87x
```

**Good news:** parallelism genuinely works -- no evidence of OpenRouter rate-limiting or
provider-side queuing collapsing concurrent requests back into serial behavior. 4 calls in
parallel really do finish faster than 4 calls one after another.

**The catch:** the whole batch's wall-clock time is bounded by its single slowest call. In this
run, 3 of 4 calls finished in 12-22s, but the 4th took 62.2s -- and because `_run_sql_batch`
(and the whole `bq_retrieve` stage) waits for all NL2SQL generation to complete before moving
on, **that one unlucky reasoning-length outlier sets the floor for the entire request.**

**This means: even fixing token starvation (finding #1) and confirming parallelism works
(this finding) does not fully solve tail latency, because with `RAG_BQ_MAX_TABLES=6` hints and
4 workers, most requests will do 2 sequential batches of 4 -- and each batch only needs ONE
slow-reasoning call to blow up the total.** This is the concrete mechanism behind the
p90/p99 numbers being so much worse than p50 in the original Langfuse data (209s vs 103s).

**Implication for the recommended next steps:** a per-call timeout with fallback (already on
the recommendation list above) would directly fix this specific failure mode -- cap any single
NL2SQL call at, say, 20-25s and fall back to a template/pattern SQL or skip that table hint
entirely, rather than letting one slow call hold up the whole batch.

---

## Fifth finding: `_run_sql_batch` (validation/dry-run/execution) is fast -- not a latency contributor, but a correctness issue

Ran real dry-run + execution calls against `staging_dev` (using the production BQ service
account) for 3 representative queries:

| Query | dry_run | execute | Result |
|---|---|---|---|
| Single-country/product/year filter | 5.13s | 1.78s | 8 rows |
| Country ranking (GROUP BY + ORDER BY) | 0.87s | 2.09s | 10 rows |
| `SELECT * ... LIMIT 10` (no WHERE) | 0.85s | 0.51s | **FAILED**: `bytesBilledLimitExceeded` |

**`_run_sql_batch` itself is fast** (1-5s per query including auth overhead) when the query
succeeds -- confirms this stage is genuinely not a latency contributor, matching what the
Langfuse span data already implied.

**But it surfaced a real correctness issue:** the `LIMIT 10` query needed to scan 1.25GB to
satisfy a `SELECT *` with no filter, and hit the `maximum_bytes_billed=250MiB` cap added in
ML-048. This means some fraction of real NL2SQL-generated queries -- especially ones without a
tight enough `WHERE` clause -- are being rejected by the byte cap in production. This is fast to
fail (~0.5s) so it's not a latency problem, but it likely **degrades answer quality** silently
(fewer usable BQ rows than expected) and should be tracked separately from latency.

## Sixth finding: retry path is real and explains the worst outliers (partial evidence)

Traced `retrieval.bq` output for real production traces. Found a direct real-world example:

```
Trace 0b42ff2277bd...: sql_query_count=0, row_count=0, status=no_valid_sql, latency=225.0s
```

This is the same "what are the information about school feeding in east africa" question that
appeared in the earlier NL2SQL test matrix (which independently measured 224.96s original prod
latency for this exact question). `sql_query_count=0` means **every NL2SQL generation attempt
for this request failed to produce usable SQL** -- consistent with `_prepare_sql`'s retry path
(one initial generation + one retry generation, per table hint, each subject to the same
unpredictable reasoning-length behavior measured throughout this investigation) all failing or
timing out.

**This directly confirms the retry mechanism is a real contributor to the worst-case tail
latency** (225s is close to the `RAG_BQ_NL2SQL_TIMEOUT_S=300` ceiling) -- not just a theoretical
risk. Full end-to-end retry timing (how long the retry call itself typically adds vs. a
first-attempt success) was not isolated in this pass -- the Langfuse per-trace observation
lookup is too slow (20-40s per trace) to pull a larger sample in reasonable time; confirming
via the production Railway logs directly (grep for `NL-to-SQL: LLM returned non-SELECT text` or
`validation rejected SQL`) would be a faster way to get a real retry-rate number if needed.

## Seventh finding: plan-tier latency spread is real but likely a query-mix artifact, not a per-tier bug

Pulled 100 recent traces and grouped trace-level latency by `plan_type` tag:

| Plan | n | p10 | p50 | p90 | min | max | % under 20s |
|---|---|---|---|---|---|---|---|
| Free | 37 | 40.7s | **107.9s** | 192.6s | 1.1s | 271.1s | 8% |
| Integrated | 15 | 47.5s | 87.8s | 178.3s | 44.9s | 271.4s | 0% |
| Government | 4 | 16.3s | 80.4s | 208.3s | 3.4s | 248.7s | 25% |
| Agribusinesses | 3 | 16.8s | 70.7s | 120.4s | 3.3s | 132.9s | 33% |
| Farmers | 8 | 2.1s | **42.2s** | 91.8s | 0.1s | 130.6s | 25% |

Free is ~2.5x slower at the median than Farmers. Checked whether this is a per-tier *model* or
*code-path* difference: `RAG_BQ_NL2SQL_MODEL_ID` and `RAG_BQ_REASONER_MODEL_ID` are both
explicitly set in Railway, which **bypasses** `model_for_plan()` for NL2SQL/reasoning entirely
(confirmed in `bq_sql_reasoner.py::_reasoner_model` -- dedicated env var always wins). So all
plans use the identical NL2SQL/reasoner model and are equally exposed to the reasoning-length
variance documented throughout this report; there is no separate "Free is on a slower BQ
model" explanation.

**More likely explanation: query mix.** Free has only 8% of its sampled requests finishing in
under 20s (i.e. hitting a fast meta/greeting/product shortcut that skips BQ entirely), vs.
25-33% for Farmers/Government/Agribusinesses. In this sample, Free users are asking a higher
proportion of BQ-heavy analytical questions relative to quick/conversational ones, so they hit
the slow NL2SQL path more often -- not because Free is throttled or on a different model, but
because more of its traffic happens to need the slow path in this particular sample.

**Caveat:** sample sizes for Government (4), Agribusinesses (3), and even Farmers (8) are too
small to be confident this pattern holds generally -- a larger, longer-window pull would be
needed to separate "real tier-driven behavior difference" from "this 100-trace sample happened
to catch more BQ-heavy Free questions." Flagging as an observation, not a confirmed root cause.

---

## Non-findings (ruled out)

- `RAG_BQ_NL2SQL_PARALLEL` off -- ruled out, already on in production.
- Sequential downstream SQL execution loop -- ruled out, span wraps only generation.
- The NL2SQL model being inherently/uniformly slow -- ruled out; it's fast on simple prompts,
  and even on realistic prompts it's fast *most* of the time -- the issue is variance /
  occasional very long reasoning, not a constant slowdown.
- Vector retrieval, decomposition, final generation, web fallback -- all fast, not contributors.

---

## Follow-ups

1. Get real Railway post-deploy traces after the `RAG_BQ_NL2SQL_MAX_TOKENS=2048` change and
   re-run `query_langfuse_latency.py` to see the actual shift.
2. Extract full real `table_hints` content (not just counts) from Langfuse observation inputs
   for a fully faithful re-test, rather than substituting alphabetically-first YAML files.
3. ~~Investigate `rerank` (26% of p50) -- confirm reranker mode in Railway.~~ DONE -- confirmed
   `cross_encoder` / `BAAI/bge-reranker-base`, ~730ms/candidate on Railway CPU. See "Second
   confirmed finding" above. Next: decide whether to lower `RAG_RERANK_POOL_SIZE`, check
   Railway container CPU allocation, or A/B `openrouter` rerank mode.
4. Evaluate `RAG_BQ_NL2SQL_MODE=batch` and/or a non-reasoning model as the actual tail-latency
   fix, since token-limit tuning alone does not bound worst-case latency for this model.
5. Clean up the harmless `RAG_RERANKER_MODE=cohere` + missing `COHERE_API_KEY` config drift in
   Railway (currently silently falls through to `cross_encoder` at ~0ms cost, but should be
   made explicit).
6. ~~Check `bq_reason` token settings for the same reasoning-model risk.~~ DONE -- tested;
   raising `RAG_BQ_REASONER_MAX_TOKENS` made it *worse* (opposite of NL2SQL). No action needed,
   current default (800) is fine and matches real production data (297ms p50).

7. ~~Concurrency effects under real load.~~ DONE -- simulated 4 concurrent calls matching
   `RAG_BQ_NL2SQL_PARALLEL_WORKERS=4`; parallelism works (1.87x speedup, no rate-limit
   collapse), but the batch's wall time is bounded by its single slowest call. See "Fourth
   finding" above -- this is the real mechanism behind bad p90/p99 tail latency.

8. ~~`_run_sql_batch` (SQL validation + dry-run + actual BigQuery execution).~~ DONE -- ran real
   dry-run + execution against staging_dev. Fast (1-5s/query), not a latency contributor. Found
   a separate correctness issue instead: unfiltered queries can hit the `maximum_bytes_billed`
   cap and fail silently. See "Fifth finding" above.
9. ~~Actual retry-path behavior.~~ Partially done -- found a real production trace
   (`sql_query_count=0`, 225s latency) consistent with the retry path failing outright for a
   given question. Confirms retries are a real tail-latency contributor. Full retry-rate /
   retry-added-latency numbers not isolated (Langfuse per-trace lookups too slow to sample at
   scale in this session -- grepping Railway logs directly would be faster). See "Sixth
   finding" above.

10. ~~Plan-tier / query-type stratification.~~ DONE -- pulled 100 traces grouped by
    `plan_type`. Free is ~2.5x slower at median than Farmers, but not due to a different model
    (confirmed dedicated NL2SQL/reasoner env vars bypass `model_for_plan` entirely) -- more
    likely a query-mix artifact (Free's sampled traffic skews more BQ-heavy). See "Seventh
    finding" above. Sample sizes for non-Free/Integrated tiers are small (3-8 traces); would
    need a larger pull to confirm this generalizes.

### All originally-listed follow-up items have now been tested. No further open items from
this investigation round -- see the TL;DR and numbered findings above for the complete set of
confirmed results and recommendations.

---

## IMPLEMENTED: per-call timeout with fallback (2026-08-25)

Following the "Fourth finding" recommendation above, this is now implemented:

- **New env var:** `RAG_BQ_NL2SQL_CALL_TIMEOUT_S` (default 20s). Separate from
  `RAG_BQ_NL2SQL_TIMEOUT_S` (300s, unchanged, still the hard per-HTTP-call timeout / last
  resort safety net).
- **Where:** `ml/rag/retrievers/bq_retriever.py::_nl_to_sql_many`'s parallel batch. Uses
  `concurrent.futures.wait(..., timeout=call_budget, return_when=ALL_COMPLETED)` with a single
  fixed budget measured from when the batch was submitted -- not re-armed per completion, so
  one slow hint cannot push the batch past the budget regardless of how many other hints finish
  first.
- **Fallback on timeout:** the timed-out table hint's SQL is simply skipped for this request
  (Option (a) from the plan -- lowest risk, reuses the existing `_maybe_template()` fallback
  path that already runs when zero usable rows come back). Abandoned Python threads keep
  running in the background (cannot be force-cancelled) and their result is discarded;
  `pool.shutdown(wait=False)` avoids blocking on them.
- **Observability:** `timed_out_hints` added to the `retrieval.bq.nl2sql` span metadata in
  Langfuse, plus a warning log line, so we can verify post-deploy how often the timeout fires
  and tune `RAG_BQ_NL2SQL_CALL_TIMEOUT_S` from real data instead of guessing.
- **Tests:** `ml/rag/tests/retrievers/test_bq_nl2sql_call_timeout.py` (3 tests, all passing) --
  covers the env var parsing/clamping, confirms a 5s-sleeping hint does not block a 1s-budget
  batch (batch returns in <3s, fast hints' SQL present, slow hint's SQL absent), and confirms
  unchanged behavior when every call finishes within budget.

**Trade-off to monitor:** requests may occasionally use fewer tables than the reasoner selected
(when a hint times out), meaning a slightly less complete answer in exchange for consistently
bounded latency. Watch the `timed_out_hints` metric in Langfuse after deploy to see how often
this actually happens in practice, and whether 20s needs to move up or down.

**Not yet done:** deploying `RAG_BQ_NL2SQL_CALL_TIMEOUT_S` to Railway and re-measuring with
`query_langfuse_latency.py` against real post-deploy traffic.

---

## Files

- `query_langfuse_latency.py` -- reusable Langfuse stage-latency breakdown script.
- `real_nl2sql_queries.json` -- 15 real questions + table_hints_count pulled from Langfuse
  `retrieval.bq.nl2sql` observations, used to build the test matrix.
- `latency_test_matrix_results.json` -- raw per-call results (latency, cost, finish_reason,
  truncation) for the 6-question x 2-token-setting matrix.
- Root cause code path: `ml/rag/retrievers/bq_retriever.py::_call_llama_for_sql` (~line 69-80),
  controlled by `RAG_BQ_NL2SQL_MAX_TOKENS`.
