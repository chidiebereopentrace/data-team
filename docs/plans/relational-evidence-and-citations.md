# Relational evidence and concise warehouse citations

## Requirements

- Public warehouse citations identify the institution and indicator, grouping repeated rows across places, products and years. Keep distinct indicators and documentary sources separate. Remap duplicate inline markers consistently in API responses, displayed sources and exported reports. Row-level evidence remains available internally for numerical grounding and ACF scoring.
- Agricultural labour productivity, crop yield, physical production, employment shares and income are distinct measures. Bind sector productivity to the mart's existing `Agriculture value added per worker (constant 2015 US$)` indicator, `sex = Total`, `unit = USD`. Land/crop productivity still resolves to yield. A multi-indicator question must compile separate correctly filtered queries even when indicators share one table.
- Every explicit document requirement participates in admission, reranking, coverage and generation for reports as well as comparisons. Keep relevant household studies alongside verified warehouse observations. Publication dates and observation dates remain distinct; older bibliographic references do not by themselves invalidate an in-period study. Evidence between compared windows can explain changes without being mistaken for an endpoint observation.
- Do not infer farmer income from employment shares or women’s share of agricultural workers from agriculture’s share of female employment. The answer must identify the denominator and state genuine evidence gaps.
- Preserve place/time constraints and relevance checks; report document search failures separately from no matching passage. Use the existing BigQuery, Qdrant and OpenRouter paths.

## Regression checks

Reproduce the supplied Nigeria/Ghana/Uganda 2015–2024 question at contract compilation, SQL planning, document admission, reranking and final citation presentation. Check distinct source/indicator identities, marker remapping, all verified observations, and land versus labour productivity. Extend the existing context and generation checks to expect grouped citations.

The live warehouse check verifies two queries and 54 returned rows: 27 agricultural employment-share and 27 agricultural value-added-per-worker observations. Live document retrieval and end-to-end generation are verified separately from deterministic passage replays.

## Live warehouse check — 2026-11-20

The Nigeria, Ghana, and Uganda 2015–2024 question was sent to the restarted local API at `POST /query/integrated` (`http://127.0.0.1:7860`), which is the API the inspector at `http://127.0.0.1:8503/` calls. Two bind-compiler queries against `fct_employment` completed with status `ok`: 27 rows for agricultural employment share and 27 rows for agriculture value added per worker. The public citations are two `structured_data` entries, one per indicator: `FAOSTAT — Share of employment in agriculture in total employment - ILO Modelled Estimates` and `FAOSTAT — Agriculture value added per worker (constant 2015 US$)`. Years and countries share those citations. The verified table contains 2015–2023 for both measures in all three countries. 2024 is absent from the warehouse, and the answer says those years were not verified. This request admitted no document passages; `news_data` and `public_reports` are unhealthy on the current ingest contract, and the answer says a document explanation could not be verified. That document gap is outside this warehouse check.
