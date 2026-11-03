---
type: architecture-session
engagement: as_is_review
slug: whole-system-as-is
started: 2026-10-14
status: complete
decision_map: ""
---

# Architecture Session: Whole-system as-is

## Goal

Judge the data-team monorepo as it runs, not the uncommitted geography patch. Name the live product, the words that collide, and the docs that still describe other systems.

## Engagement

- Type: as_is_review
- Roles run: context-and-constraints, domain-architecture, application-architecture, data-architecture, integration-architecture

## Timeline

- 22:51 — Opened session from the whole-project judgment
- 22:55 — Context and constraints
- 22:55 — Domain architecture
- 22:55 — Application seams
- 22:55 — Data and integration
- 22:55 — Synthesis, glossary, ADRs, vault corrections

## Role outputs

### Context and constraints

Ask ADZA is the product that answers a question. Document evidence comes from six Qdrant corpora. Fact evidence comes from BigQuery `mart_dev`, bind-first, with free SQL only when a bind is incomplete. The graph decomposes, runs both legs, merges, reranks, and generates. Serving is FastAPI and Streamlit, toward Railway.

`data-eng/` lands sources with Airbyte and models `landing` → `raw_dev` → `staging_dev` → `intermediate_dev` → `mart_dev`. Feature-store, training, and evaluation packages do not change an answer.

There is no root `CONTEXT.md` and there were no ADRs. The vault glossary is closer to the code than the repo docs, and it still defines `plan_type` as a reasoner classification. In code that field is the access tier.

Constraints for this pass: record language and the two decisions the code already made. Do not refactor `graph.py`, delete stubs, or ADR the always-on dual leg.

### Domain architecture

Bounded contexts:

- **Warehouse** — land and model African agriculture facts. Owns Raw and the Mart.
- **Retrieval** — turn a Question into Document evidence and Fact evidence, then a Gap when neither holds.
- **Answer** — write in a Persona, under a Plan, from the evidence of this turn.
- **Ingest (documents)** — Drive and news mining into the corpora. Separate from Airbyte.

Core domain is Retrieval plus the Mart contract. Packaging and stubs are supporting.

Language that breaks design:

- **Plan** is the access tier, the Generation plan, and the Bind plan. The vault used one name for the last of those.
- **Place** is Question place, Home place, a UI override, warehouse `place_scope` / `data_level`, and Qdrant `geo_scope`.
- **Layer** is landing, raw, staging, intermediate, and mart, and also bronze, silver, and gold. The RAG architecture doc forbids querying silver/gold in the same section that queries the mart.
- **ACF** is warehouse columns and a scoring pass over cited evidence.

The Albania gap is this language at question scope: Plan was used as Place, and Home place was allowed to sit outside Africa.

### Application architecture

`ml-eng/ml/rag/chatbot/graph.py` is the product. Its interface is the whole graph state. Callers cannot ask what was retrieved without knowing that bag. That seam is too wide. Leave it for a later pass.

The other failure is pass-through. Feature and training packages, the legacy SQL compiler, and the root diagram's training-to-serving box add surface without behaviour.

What is deep enough to protect: bind-first fact retrieval, warehouse fail-closed (skip facts, still return documents), the indicator-class spine, the production-versus-yield ban, source identity, and the Africa gazetteer.

Open question, not a decision: both legs always run. Routing research (WWW 2026, Learning to Route) finds that a naive blend of documents and a database adds noise and cost. VLDB hybrid-search results find a weak path can drag a fused ranking down. The current choice keeps a literature-only or warehouse-only answer from being dropped. `select_corpora` routes among document corpora. It does not choose whether facts are needed.

### Data architecture

Live answers read the Mart. Raw is for replay. Databricks medallion practice and dbt's mart layer say the same thing: consumers read the refined contract.

`data-eng/docs/GLOBAL_ARCHITECTURE.md` (2026-06-27) still says bronze is the ML contract and that RAG queries `raw_dev`. Code and tests query `mart_dev`.

Three inventories disagree: the analyst "live" set, the dbt SQL count, and the YAML bind cards. Nine of fifteen indicator-class schema cards are stubs. The Airflow dbt DAG is an empty operator, so the medallion is not a schedule. OTA insights can be an empty collection while diagrams show six corpora.

### Integration architecture

Two ingestion worlds do not share a document term: Airbyte into BigQuery, and Drive plus news mining into Qdrant.

`data-eng/docs/ARCHITECTURE.md` still points retrieval at Vertex AI Vector Search and serving at a Prediction API. Retrieval is Qdrant. `ml-eng/README.md` still names one collection, `opentrace_rag`. `data-eng/README.md` says ML lives at repo-root `ml/`. Source of truth is `ml-eng/ml/`.

`ml-eng/ml/rag/README.md` draws the warehouse leg after the document leg. The architecture doc and the graph run them together, then merge.

The vault at `af6ca53` already recorded the bronze-versus-mart split, the empty DAG, the stubs, and the OTA gap. The cloned repo still describes the older system.

## Cross-cutting synthesis

- Risks: a new engineer follows `GLOBAL_ARCHITECTURE.md` or the "staging-only / never silver/gold" sentence and queries the wrong layer; `plan_type` in the vault glossary gets wired into SQL routing; the next profile or layer bug looks like a parser bug.
- Open questions: whether both legs should stay peers for every question; which mart inventory is the consumer contract (analyst live set, dbt SQL, YAML cards); which stub packages are scheduled for deletion.
- Recommended ADRs: answers read the mart; Plan does not replace Question place. Do not ADR the dual leg.
- CONTEXT.md / project-context updates: root glossary for Question, Place, Plan, Persona, Generation plan, Bind plan, Document evidence, Fact evidence, Mart, Raw, Indicator class, Source, Open-ended time, Gap.

## Links

- Decision map: none
- ADRs: `docs/adr/0001-answers-read-the-mart.md`, `docs/adr/0002-plan-does-not-replace-question-place.md`
- Vault: [[Live Retrieval Path]], [[Glossary]], [[Handoff Failure Modes]], personal note `2026-10-14 Whole Project Architecture Judgment`
- HTML as-is report: not run
- Context canvas: not refreshed

## Next step

A separate doc pass on `data-eng/docs/GLOBAL_ARCHITECTURE.md`, `data-eng/docs/ARCHITECTURE.md`, `ml-eng/README.md`, and `ml-eng/ml/rag/ARCHITECTURE.md`. Not part of this session.
