# Bibliographic Metadata Gap Report

**Sprint 1, Week 3 · TASKS.md §4 (citation enrichment)**
**Date run:** 2026-07-15
**Command:**

```
# from data-team/ml-eng, with QDRANT_URL / QDRANT_API_KEY in config/.env
python -m ml.rag.find_missing_bibliography --samples 10
```

## Purpose

Automates the *identification* half of TASKS.md §4 — locating the records that
lack a clickable link (DOI/URL) or core citation fields — so the manual DOI/URI
lookup can be targeted at the highest-impact papers. The actual lookup + Qdrant
payload update remains a manual data-curation step.

## Results (all research collections scanned)

| Collection | Points | Missing link | Missing authors | Missing year | Missing title |
|---|---:|---:|---:|---:|---:|
| `academic_papers` | 32,315 | 18,699 (57.9%) | 30,082 (93.1%) | 7,039 (21.8%) | 0 (0.0%) |
| `policies` | 1,734 | 1,332 (76.8%) | 1,645 (94.9%) | 415 (23.9%) | 0 (0.0%) |
| `news_public_reports` | 3,568 | 2,300 (64.5%) | 3,505 (98.2%) | 1,007 (28.2%) | 0 (0.0%) |
| **Total** | **37,617** | **22,331 (59.4%)** | — | — | — |

## Key takeaways

- **Titles are essentially complete (0% missing)** across all three collections —
  so the fallback render (`format_academic_citation` → `Title (Year)`) always has
  something to show. Citations never break; the gap is *clickability + authorship*.
- **~59% of records lack a clickable link** (DOI or URL). This is the primary
  enrichment target — links are what users click through to verify a source.
- **Authors are missing on the vast majority** (93–98%). Year is missing on ~22–28%.
- Note: the earlier "200–600 records" estimate in TASKS.md substantially
  **understated** the gap — the real figure is ~22.3k points missing a link.
  Full manual enrichment of all of them is out of scope for Sprint 1; the
  practical play is to enrich the **highest-impact / most-cited** subset first
  (see priority sample below) and continue the rest post-launch (see TASKS.md
  post-launch item).

## Priority enrichment targets (sample — no link + missing author/year)

From `academic_papers`:
- WEST AFRICAN AGRICULTURE AND CLIMATE CHANGE
- Agricultural Growth Linkages in Sub-Saharan (Africa)
- Agriculture and Future Climate Dynamics in Africa: Impacts and Adaptation
- THE DIGITALISATION OF AFRICAN AGRICULTURE REPORT 2018-2019
- DEVELOPMENT OF THE CONSERVATION AGRICULTURE EQUIPMENT INDUSTRY IN SUB-SAHARAN AFRICA

From `policies`:
- RECIPES FOR SUCCESS 2: Policy Innovations to Achieve the Kampala Declaration Goals
- AGRODEP Working Paper 0010 / AGRODEP Technical Note 07

From `news_public_reports`:
- CGIAR Platform for Big Data in Agriculture — Annual Reports 2017–2020
- EXTERNAL DEVELOPMENT FINANCIAL FLOWS TO FOOD SYSTEMS

Data-quality flags seen in the sample (worth a cleanup pass during enrichment):
- Some records have mojibake / encoding artefacts in titles (e.g. `�`), suggesting
  a PDF text-extraction encoding issue upstream.
- At least one clearly off-topic record surfaced ("THE λ-INVARIANT CHANGE FOR
  ABELIAN VARIETIES…" — a pure maths paper), i.e. corpus-relevance noise the Data
  team may want to prune.

## Status vs. TASKS.md §4

- [x] Fix inline citation rendering — done (`_normalize_inline_citations`, `_strip_model_sources_appendix`).
- [x] Clickable links whenever DOI/URI exists — done (`_citation_url()` covers all source types).
- [x] Fallback render (title + author + year) when link missing — done (`format_academic_citation`).
- [x] **Identify** the records missing metadata — **done Jul 15** (this report + `find_missing_bibliography.py`).
- [x] **Automated enrichment script built** (ML-046) — `ml/rag/enrich_bibliography.py`
  queries CrossRef API by title, applies fuzzy-match threshold (0.85 Jaccard),
  writes `doi`, `url`, `authors`, `publication_year` to Qdrant. Resumable via
  checkpoint file. Dry-run by default; `--apply` to write.

  Smoke test results (3 titles from priority sample above):
  - "Agricultural Growth Linkages in Sub-Saharan Africa" → sim=1.00 → DOI `10.2499/0896291103rr107` ✅
  - "West African Agriculture and Climate Change" → sim=0.67 → rejected (below 0.85) ✅ correct
  - "THE DIGITALISATION OF AFRICAN AGRICULTURE REPORT" → sim=0.33 → rejected ✅ correct

  **To enrich the full corpus:**
  ```bash
  # Dry run first (safe — no Qdrant writes):
  python -m ml.rag.enrich_bibliography --dry-run

  # Apply (starts with academic_papers — ~32k points, ~32k CrossRef calls at 1/s = ~9h):
  python -m ml.rag.enrich_bibliography --apply --collections academic_papers

  # Resumable — checkpoint saves progress every 50 points:
  python -m ml.rag.enrich_bibliography --apply --checkpoint data/local/enrich_checkpoint.json
  ```
- [ ] **Run enrichment on full corpus** — 22,331 points missing DOI; `enrich_bibliography.py` ready.
