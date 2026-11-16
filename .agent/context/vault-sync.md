# Vault sync

last_synced_commit: c730dd1
synced_at: 2026-11-16
baseline_note: |
  Synced 9a65f1d → c730dd1 (ML-072 RAG upgrades, DPL-068/067/065 ingestion/engine updates, data-eng reorg to transform-v2).
  vault-stale: 0 stale / 0 skipped. Working tree has uncommitted changes in graph.py, corpus_catalog.py (see .gitignore status).
  
## Session Scope

- Range: 9a65f1d → c730dd1 (~62 commits)
- Focus areas: ML-072 RAG dual-leg upgrades (English query variant), DPL-068/067/065 ingestion engine standardization, transform-v2 architecture migration
  
## Key Themes

### 1. ML-072 RAG System Upgrades (13 commits)
- Dual-leg retrieval: BQ + web fallback with bilingual English query variant
- Evidence preservation across retrieval retries
- Production vocabulary and greeting checks restored

### 2. Ingestion Engine Standardization (DPL-065/067/068, ~14 commits)
- Beefing up price SQL retrieval capabilities  
- RAG system auth setup for protected sources
- Global dashboard for data profiles

### 3. Data-eng Architecture Reorg (transform-v2)
- Airbyte tree archived; Python ingestion writes landing_dev/raw_dev
- dbt transform: raw_dev → mart_dev without rewriting existing SQL
- New deep modules: ingestion/, dbt/ (versioned), local/, infra/, ci/

### 4. New Agent Skills Added
- agronomy-rag, corpus-curator, multilingual-qa, rag-verifier

## Uncommitted Changes (git status --porcelain)

```
?? analyze_graph_bugs.py
 M data-eng/ingestion/notebooks/wfp_hbx_46_data_upload.ipynb
```

## Notes Updated / Created / Stale

- **Updated:** None directly in this session (no prior notes referencing these commits found)
- **Created:** Session note below
- **Stale:** 0 vault notes flagged stale

## Open Questions

1. How are ML-072 RAG upgrades integrated with existing retrieval paths?
2. What is the deployment timeline for transform-v2 vs legacy layers?
3. Are there any breaking changes in the ingestion API contracts?

## Links

- ADRs: `data-eng/docs/adr/` (transform-v2 contract, manifest-ingestion)
- CONTEXT: `data-eng/CONTEXT.md`