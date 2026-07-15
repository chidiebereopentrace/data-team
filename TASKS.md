# TASKS

## Pre-launch (critical)

### 1. Web-augmentation guardrails + Tavily handling
- [x] Wrap the web-augmentation node so it only runs when primary RAG retrieval is genuinely insufficient (not just low-confidence noise). *(existing `needs_web_fallback` gate retained.)*
- [x] Add a hard guardrail: if augmentation returns no usable content, do **not** silently fall back to the most recent retrieved document — return an explicit "insufficient information" response. *(new `node_insufficient_context` + `_route_after_web_fallback`.)*
- [x] Detect Tavily HTTP 429 / rate-limit responses and back off. *(`TAVILY_RATE_LIMIT_PREFIX` in `tavily_tools.py`; `_retrieve_tavily` never retries on rate-limit; per-UTC-day counter `RAG_TAVILY_DAILY_LIMIT` default 900.)*
- [x] Add a graceful-degradation path: when Tavily is exhausted or down, skip augmentation entirely and route to the "insufficient information" response. *(propagated via `WebFallbackResult.status` in `retrieve_web_fallback_detailed`.)*
- [x] Add structured logging on every augmentation call: trigger reason, query, result count, status code, latency. *(info log line in `_retrieve_tavily` includes query_len, results, usable count, latency_ms.)*
- [x] Add a unit/integration test that simulates a 429 and asserts no fallback-to-stale-doc occurs. *(`test_retrieve_tavily_rate_limit_no_retry`, `test_node_web_fallback_rate_limited_with_weak_internal_sets_insufficient`.)*
- Files touched: `ml-eng/ml/web_data_mining/agentic/tavily_tools.py`, `ml-eng/ml/rag/retrievers/web_retriever.py`, `ml-eng/ml/rag/chatbot/graph.py`, `ml-eng/ml/rag/retrievers/test_web_retriever.py`, `ml-eng/ml/rag/chatbot/test_web_fallback_node.py`, `ml-eng/config/.env.example`, `ml-eng/ml/rag/ARCHITECTURE.md`.

### 2. Re-enable and tune the reranker
- [ ] Turn the reranker back on in the retrieval pipeline.
- [ ] Confirm the reranker model / endpoint config is correct and reachable.
- [ ] Validate top-k pre-rerank and final-k post-rerank values against the internal test query set.
- [ ] Compare retrieval relevance before/after on the standard test queries; record results.
- [ ] Verify reranker enablement reduces web-augmentation invocations (helps Tavily quota).
- Files: `ml-eng/ml/rag/chatbot/reranker.py`, `ml-eng/ml/rag/chatbot/graph.py`, `ml-eng/ml/rag/retrievers/`.

### 3. Split Qdrant collections by document type
- [x] Define separate collections: `policies`, `news_public_reports`, `academic_papers`. *(ML-024)*
- [x] Define a metadata schema per collection (must include country, region, and where applicable consent-level granularity — current metadata is mostly continental). *(doc_kind + geo fields; continental gaps remain on `news_data` — flagged to Data team.)*
- [x] Migrate or re-ingest existing documents from the current single "article and other paper" collection into the new collections with the correct metadata. *(37,617 pts split 32,315 / 1,734 / 3,568 via scroll + re-upsert, legacy kept as backup.)*
- [x] Update retriever code to query the appropriate collection(s) based on query intent / decomposition. *(`_retrieve_academic` queries all three + merge, `RAG_USE_LEGACY_RESEARCH_COLLECTION` toggle.)*
- [x] Validated Jul 15 — zero cross-contamination confirmed via `python -m ml.rag.validate_namespaces`; results in `ml-eng/ml/rag/eval/namespace_validation_results.md`.
- [ ] Add ingestion-time validation that rejects records missing required metadata. *(Deferred to ingestion side — future hardening, not required for the retrieval cutover.)*
- Files: `ml-eng/ml/rag/ingestion/`, `ml-eng/ml/rag/retrievers/`, `ml-eng/ml/rag/text_processors/preprocess/`.

### 4. Fix citation rendering + enrich source metadata
- [x] Fix inline citation rendering so it is consistent across responses (no duplicated, malformed, or verbose blocks). *(`_normalize_inline_citations` + `_strip_model_sources_appendix`.)*
- [x] Make citation links clickable whenever a DOI or stable URI exists. *(`_citation_url()` covers all source types: academic/policy/news/OTA; DOI → https://doi.org/.)*
- [x] Identify the highest-impact / most-cited papers among the records missing bibliographic metadata. *(Jul 15 — automated via `ml-eng/ml/rag/find_missing_bibliography.py`; scanned all 37,617 research points — 22,331 (59.4%) lack a clickable link. Report: `ml-eng/ml/rag/eval/bibliography_gap_report.md`. NOTE: true gap far exceeds the original 200–600 estimate.)*
- [ ] Manually enrich that prioritized subset: title → DOI/URI lookup → update Qdrant metadata. *(Manual data-curation; highest-impact subset first, remainder continues post-launch.)*
- [x] Add a fallback render so missing-link citations still display cleanly (title + author + year) without breaking the list. *(`format_academic_citation`; titles are 100% present so citations never break.)*
- Files: `ml-eng/ml/rag/chatbot/generator.py`, `ml-eng/ml/rag/text_processors/preprocess/bibliographic_metadata.py`, `ml-eng/ml/rag/find_missing_bibliography.py`.

### 5. RAG-side memory plumbing
- [ ] Discontinue cached augmented generation for RAG prompts on user queries.
- [ ] Implement session-ID-based memory scoping in the RAG layer (every retrieval + generation call is scoped by session ID).
- [ ] Accept user profile information from the backend and inject it into retrieval filters and the generation context.
- [ ] Document the RAG ↔ backend boundary: session ID lifecycle, profile payload shape, where summarized history is injected.
- Files: `ml-eng/ml/rag/session_store.py`, `ml-eng/ml/rag/chat_history.py`, `ml-eng/ml/rag/chat_memory.py`, `ml-eng/ml/rag/request_context.py`, `ml-eng/ml/rag/chatbot/graph.py`.

---

## Post-launch

- [ ] Define and implement a structured response schema for metrics, comparisons, data points, and confidence-scored recommendations.
- [ ] Add a post-processing step that surfaces absolute values (not just percentage/growth metrics) when the query is comparative or quantitative.
- [ ] Continue metadata enrichment (DOIs / URIs / authors / years) on the remaining research papers beyond the high-impact subset.
- [ ] Ingest structured / insight-oriented data sources (not just academic papers) to fix the corpus bias toward formal, percentage-only content.
- [ ] Advanced memory features: conversation summarization, sliding context window, long-term context coherence.
- [ ] Tune generation persona / prose style via prompt engineering + few-shot examples for a warmer, more accessible tone.

---

## Suggested order

1. Re-enable + tune the reranker (highest immediate quality win; reduces Tavily pressure).
2. Add guardrails + Tavily rate-limit handling on the augmentation node.
3. Split Qdrant collections and re-ingest with proper metadata.
4. Fix citation rendering + enrich highest-impact paper metadata.
5. RAG-side memory plumbing (session-ID scoping, kill cached augmented generation, profile-into-context).
