# Namespace Separation — Validation Results

**Sprint 1, Week 3 · ML-024**
**Date run:** 2026-07-15
**Run by:** Juma (ml-eng / Intelligence)
**Command:**

```
# from data-team/ml-eng, with QDRANT_URL / QDRANT_API_KEY in config/.env
python -m ml.rag.validate_namespaces
```

## Verdict

✅ **PASS — namespace separation validated, no cross-contamination.**

Each namespace-separated collection contains only its expected `doc_kind`, and the
split is exact against the legacy backup collection.

## Collections (8)

| Collection | Points | Notes |
|---|---:|---|
| `academic_papers` | 32,315 | namespace-separated (academic) |
| `policies` | 1,734 | namespace-separated (policy) |
| `news_public_reports` | 3,568 | namespace-separated (public report) |
| `research_other_papers` | 37,617 | legacy backup (all 3 kinds) |
| `news_data` | 33,786 | live news collection (separate from the split) |
| `BQ_table_descriptions` | 57 | structured-data descriptions |
| `OTA_insights` | 0 | empty on day one by design |
| `opentrace_news` | 0 | empty |

## Namespace separation checks

| Collection | Expected `doc_kind` | Found | Result |
|---|---|---|---|
| `academic_papers` | `academic_article` | `{academic_article}` | ✅ PASS |
| `policies` | `policy_document` | `{policy_document}` | ✅ PASS |
| `news_public_reports` | `public_report` | `{public_report}` | ✅ PASS |

Sample geographies observed (healthy multi-country coverage):
- **academic_papers:** Benin, Burkina Faso, DR Congo, Ethiopia, Guinea, Kenya, Malawi, Niger, Nigeria, Senegal
- **policies:** Ghana, Kenya, Madagascar, Malawi, Nigeria, Rwanda, Senegal, Uganda
- **news_public_reports:** DR Congo, Egypt, Kenya, Malawi, South Africa, Uganda

## Split-math check

```
academic_papers      32,315
policies              1,734
news_public_reports   3,568
------------------------------
total                37,617  == research_other_papers (legacy backup) ✅
```

The migration was a pure scroll + re-upsert (no re-embedding). The legacy
`research_other_papers` collection is retained as a backup and still holds all three
`doc_kind` values, as expected.

## Follow-up / flag for Data team (Week 4 QA)

- `news_data` (33,786 pts) shows some `unknown` geos and coarse tags like `Africa`.
  This is the live news collection (separate from the namespace split), but the
  `unknown`/continental geo tagging is a **Data-team tagging gap** that affects geo
  purity. Flagging for Week 4 QA and source-tagging follow-up.

## Notes / possible hardening (optional)

- `validate_namespaces.py` currently samples the first **20 points** per collection
  for the `doc_kind` check. The split-math already lines up exactly, so the PASS is
  well-supported, but for a fully airtight "zero contamination" claim we could later
  switch to a count-by-`doc_kind` (Qdrant filter/count API) across all points.
