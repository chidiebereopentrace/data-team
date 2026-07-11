# Sprint 1 — ML/AI Engineering (Intelligence) Plan

> Source: `Update Process.pptx` (OpenTrace Africa — Sprint 1 Implementation Plan, July 1 – July 31, 2026).
> Scope of this document: **ml-eng deliverables only** (the "Intelligence" track, plus RAG code that is nominally listed under "Data" in the slides but actually lives in `ml-eng/ml/rag/`).
> Companion file: [`TASKS.md`](./TASKS.md) — legacy task list; this file supersedes/extends it for the July sprint.

---

## 0. Why this sprint exists (from the deck)

Nine team members tested Ask ADZA across 72 hours in late June. The platform held technically — zero crashes, no hallucinations on specific queries, strong UI scores. But the **intelligence layer** showed clear problems that must be resolved before any external user sees the product.

Test-1 findings that are ml-eng's problem:

| Finding | What it means for ml-eng |
|---|---|
| **Data routing broken** | RAG vector DB unsegregated. African agricultural queries pulling from unrelated global academic content. → **Namespace/collection separation in Qdrant.** |
| **LLM fills gaps badly** | When ADZA has no data it synthesises academic prose instead of returning a clean, structured gap acknowledgement. → **Lower temperature + rewrite no-data fallback prompt.** |
| **ACF signals invisible** | 5 of 9 testers did not notice the confidence signals at all. → **Surface ACF band + score + plain-language note by default on every response.** |
| **Session bleed** | Follow-up queries mixing context. → **Session context isolation (session-ID scoping).** |
| **4 UI bugs** | Credit visibility, category count, mid-conversation switching, query persistence. → ❌ **Frontend, not ml-eng.** |

**Hard gate:** Zero open critical/high issues from internal test 1 by **July 31**. Internal test 2 begins **August 1**.

---

## 1. What every department delivers by July 31 (context)

| Department | Deliverables |
|---|---|
| Data | Intake 2× pre-sprint volume; all sources tagged by geography + time; RAG namespace separation live and validated; full data audit doc |
| **Intelligence (ml-eng — us)** | **LLM temperature lowered; fallback prompt rewritten; ACF signals visible by default; session context isolation; 4 UI bugs fixed¹; Pipeline QA sign-off** |
| Research | Data gap mapping; first reconstruction output; predictive intelligence scoping note; sprint 1 research brief |
| Business, Partnerships & Fundraising | 100+ segmented beta contacts; 4 new partnership meetings; 3 funding opp summaries; beta outreach materials |

¹ The 4 UI bugs sit under Intelligence on the slide but are frontend work — see §7 (Out of Scope).

---

## 2. Ml-eng deliverables — grouped and cross-referenced

Each item shows: (a) what the slide says, (b) the file(s) in `ml-eng/ml/**` that need to change, (c) the status of any prior related work.

### 2.1 LLM temperature + no-data fallback prompt
- [x] **LLM temperature** — Sprint 1 deliverable satisfied by code guardrails, not by a low value. ✅ *Shipped Jul 2 — after review with the team, default `RAG_GENERATE_TEMPERATURE` set to **0.7** in `generator.py` + `.env.example` so responses feel varied across similar queries. The test-1 "synthesises on gaps" concern is handled by the fallback path below (empty context → no LLM call at all), the hardened ungrounded prompt, and existing chunk filtering — so temperature can safely stay in the natural-prose range.*
  - Other call-sites unchanged at their own values (query decomposer, BQ retriever, reranker, HF check all use 0.0 as before).
  - Env override still respected (unit test `test_call_llama_temperature_env_override_respected`).
- [x] **Rewrite the no-data fallback prompt** to produce a clean, structured gap acknowledgement (not academic prose). ✅ *Shipped Jul 2 — new `_no_data_fallback_message()` in `generator.py`; routes both the empty-context branch and the hardened `RAG_ALLOW_UNGROUNDED=on` branch through it. Every fallback carries an explicit `ACF: no evidence` marker so the confidence signal is never invisible on this path.*
- [x] **Source purity enforcement** on geographic + temporal query metadata — ✅ *`fix(ML-019)`, merged: geo-conflicting chunks dropped before generation + thin-context threshold.*
  - Files: `ml/rag/chatbot/generator.py`, `ml/rag/chatbot/graph.py`.

### 2.2 ACF confidence signals visible by default
- [x] Every response returns `{ band, score, note }` in the API schema — **shipped Jul 11** via new `ACFSignal` Pydantic model in `api_schemas.py` and `acf` field on `QueryResponse`.
  - New module: `ml/rag/chatbot/acf.py` — standalone `compute_acf()` using weighted composite of chunk count (0.30), average relevance (0.40), source diversity (0.20), and internal-only bonus (0.10).
  - 4 bands: HIGH (≥0.75), MEDIUM (≥0.45), LOW (≥0.10), NO_EVIDENCE (<0.10) — all thresholds + copy overridable via env vars.
  - Wired into all 3 graph generate nodes (`node_generate`, `node_generate_meta`, `node_generate_product`).
  - Meta/product queries → ACF HIGH (curated knowledge base). Data queries → computed from retrieval context.
  - 13 unit tests in `ml/rag/chatbot/test_acf.py` — all passing.
  - Files: `ml/rag/chatbot/acf.py` (new), `ml/rag/api_schemas.py`, `ml/rag/chatbot/graph.py`, `ml/rag/app/api.py`, `ml/rag/chatbot/test_acf.py` (new), `ml/rag/test_api_schemas.py`.
  - Thresholds and plain-language copy are env-configurable so Chidi can adjust from the PR review without code changes.

### 2.3 Session context isolation
- [x] Retrieval + generation calls scoped by session ID; no cross-query information bleed — **shipped Jul 11**.
  - `session_id` threaded through `RAGGraphState` → `run_rag()` → API for full pipeline tracing.
  - Session store (`session_store.py`) already scopes memory by `rag:session:<sid>` — verified with 6 isolation tests.
  - New `DELETE /session/{session_id}` endpoint for explicit session cleanup (frontend calls on new-chat / logout).
  - Graph state is per-invocation (no shared mutable state) — no cross-request contamination possible.
  - Files: `ml/rag/chatbot/graph.py`, `ml/rag/app/api.py`, `ml/rag/test_session_isolation.py` (new).
- [x] Discontinue cached augmented generation for RAG prompts on user queries — **already the case**: no LLM generation results are cached across queries. Each `run_rag()` call runs the full pipeline fresh. Verified in `test_no_global_mutable_session_state`. The only caches are infrastructure-level (BQ schema, embedding models) which are session-independent.
- [ ] Accept user-profile payload from backend and inject into retrieval filters + generation context — **partially done**: `user_profile` → `geo_override` → retrieval filters is working. Full profile injection (stakeholder type, organization) deferred pending backend payload shape from Chidi (§6 open question #4).

### 2.4 RAG namespace separation (Qdrant collections) — ✅ done (ML-024)
> Listed under "Data" in the deck but the code lives in ml-eng.
- [x] Define separate collections: `academic_papers`, `policies`, `news_public_reports` — created in Qdrant, mirroring the source vector layout.
- [x] Migrate existing documents into the new collections by `doc_kind` — 37,617 pts split (32,315 / 1,734 / 3,568) via scroll + re-upsert, no re-embedding; `research_other_papers` kept as backup.
- [x] Update retriever code to query the new collections + merge — `_retrieve_academic` now queries all three, with `RAG_USE_LEGACY_RESEARCH_COLLECTION` revert toggle.
- [ ] *(Deferred to ingestion side)* Ingestion-time validation that rejects records missing required metadata — future ingestion hardening, not required for the retrieval cutover.
- Files: `ml/rag/chatbot/graph.py`, `ml/rag/scripts/migrate_research_collection.py`, `config/.env.example`.
- Prior work: `TASKS.md` §3 — ✅ core done (ML-024).

### 2.5 Reranker re-enablement (adjacent, high leverage)
> Not called out explicitly in the deck but directly supports the "direct answer before context and sources" Week 3 goal, and reduces Tavily pressure.
- [ ] Turn the reranker back on in the retrieval pipeline.
- [ ] Confirm model/endpoint config is correct and reachable.
- [ ] Validate top-k pre-rerank and final-k post-rerank against internal test query set.
- [ ] Record relevance before/after on the standard test queries.
- Files: `ml/rag/chatbot/reranker.py`, `ml/rag/chatbot/graph.py`, `ml/rag/retrievers/`.
- Prior work: `TASKS.md` §2 — ⬜ not started.

### 2.6 Citation rendering + source metadata (adjacent)
- [ ] Fix inline citation rendering — no duplicated / malformed / verbose blocks.
- [ ] Clickable links whenever DOI or stable URI exists.
- [ ] Enrich highest-impact papers among the 200–600 records missing bibliographic metadata.
- [ ] Fallback render (title + author + year) when link missing — no broken lists.
- Files: `ml/rag/chatbot/generator.py`, `ml/rag/text_processors/preprocess/bibliographic_metadata.py`.
- Prior work: `TASKS.md` §4 — ⬜ not started.

### 2.7 Pipeline QA + sign-off
- [ ] Structured test queries across **all 5 stakeholder groups** × all major geographies.
- [ ] Zero open critical issues by July 31; every test-1 issue demonstrably resolved.
- [ ] Pipeline QA sign-off document submitted to Chidi.
- Files: `ml/rag/tests/`, `ml/rag/eval/`, `test/test.jsonl`.
- Open question for Chidi: which 5 stakeholder groups + which geographies constitute "major"?

---

## 3. Week-by-week plan (ml-eng only)

### ✅ Week 1 — Jul 1 → Jul 7 · COMPLETE · Critical fixes: data routing + intelligence layer
- [x] `RAG_GENERATE_TEMPERATURE` **set to 0.7** (Jul 2) — team-agreed value for prose variety; synthesis-on-gaps risk is covered by code guardrails, not temperature.
- [x] Rewrite the no-data / insufficient-context fallback prompt → structured gap response — **shipped Jul 2** (`_no_data_fallback_message`, ACF marker included, ungrounded prompt hardened, 6 new unit tests passing).
- [x] Source purity enforcement on geographic + temporal query metadata — **shipped Jul 2** (`fix(ML-019)`, merged): geo-conflicting chunks dropped before generation + thin-context threshold that returns the structured gap message instead of padding weak context.
- [x] **Namespace separation in Qdrant** — **done Jul 7** (`feat(ML-024)`, PR open): the mixed `research_other_papers` collection split by `doc_kind` into `academic_papers` (32,315), `policies` (1,734), and `news_public_reports` (3,568). Migration was a pure scroll + re-upsert (no re-embedding); original collection kept as backup. Retriever cut over to query all three + merge, with `RAG_USE_LEGACY_RESEARCH_COLLECTION` toggle to revert instantly.
- **End-of-week 1 check:** LLM temperature adjusted ✔ Namespace separation in place ✔ Source purity live ✔
- **STATUS: ✅ Week 1 complete.** All Intelligence deliverables merged or pushed (temperature+fallback → PR #27, source purity → PR #29/ML-019, namespace separation → ML-024). Full end-to-end answer-quality validation is scheduled for Week 4 Pipeline QA on the provisioned environment.

### ✅ Week 2 — Jul 8 → Jul 14 · COMPLETE · Data quality, ACF visibility, session isolation
- [x] Surface ACF confidence signals on **every** response by default: `band`, `score`, plain-language note — **shipped Jul 11** (`acf.py`, `api_schemas.py`, `graph.py`, `app/api.py`; 13 unit tests passing).
- [x] Implement session context isolation — **shipped Jul 11**: `session_id` threaded through graph state, `DELETE /session/{session_id}` endpoint added, 6 isolation tests passing. No cross-session memory bleed.
- [x] Discontinue cached augmented generation for user queries — **already the case**: verified no LLM result caching exists. Each query runs the full pipeline fresh.
- [ ] Run structured test queries to validate the fallback-prompt rewrite from Week 1. *(Deferred to Week 4 QA — requires provisioned environment.)*
- [ ] Support geographic + temporal metadata filtering on retrieval — **already partially live** via Week 1 source purity (ML-019). Full metadata filtering works alongside Data team's tagging.
- **End-of-week 2 check:** ACF signals visible by default ✔ Session isolation live ✔
- **STATUS: ✅ Week 2 core deliverables complete.** Both end-of-week checkmarks met. Structured test queries deferred to Week 4 QA on the provisioned environment.

### Week 3 — Jul 15 → Jul 21 · Integration, response quality, data depth
- [ ] Re-enable and tune the reranker (`TASKS.md` §2). Record top-k / final-k choices.
- [ ] Validate namespace separation with test queries: confirm **zero cross-contamination** across geographies.
- [ ] Full end-to-end integration test of the query → response pipeline.
- [ ] Response quality: **direct questions must return a direct answer before context/sources.** Prompt + generator adjustments.
- [ ] Fix citation rendering + enrich highest-impact source metadata (`TASKS.md` §4).
- **End-of-week 3 check:** Reranker on ✔ Citations clean ✔ Direct-answer-first behaviour observed on test set ✔

### Week 4 — Jul 22 → Jul 31 · QA, stabilisation, sprint 1 sign-off
- [ ] Full pipeline QA: structured test queries across all 5 stakeholder groups × all major geographies.
- [ ] Close all open critical/high issues from test 1 — zero open by July 31.
- [ ] Pipeline QA sign-off document submitted to Chidi.
- [ ] Post-mortem note of any residual risk items → feeds into Sprint 2 plan.
- **July 31 gate:** Zero open critical issues ✔ Sign-off doc submitted ✔ Ready for internal test 2 on Aug 1.

---

## 4. Master checklist — mirror of slide 9 (Intelligence rows)

- [x] LLM temperature tuned; no academic synthesis on gaps *(Jul 2 — temperature at 0.7 for prose variety; gap-response enforced by fallback path, not temperature)*
- [x] No-data fallback prompt rewritten and tested *(Jul 2 — structured gap + ACF marker, 6 unit tests)*
- [x] Source purity enforced on geographic and temporal queries *(Jul 2 — ML-019, merged)*
- [x] ACF signals visible by default on every response *(Jul 11 — `acf.py` + `ACFSignal` schema, 13 unit tests, all response paths covered)*
- [x] Session context isolation, no cross-query contamination *(Jul 11 — session_id in graph state, DELETE /session endpoint, 6 isolation tests passing)*
- [ ] Pipeline QA completed
- [ ] Zero open critical issues by July 31
- [ ] Sign-off doc submitted to Chidi

Related shared items (co-owned with Data):
- [x] RAG namespace separation implemented and validated *(Jul 7 — ML-024; end-to-end answer-quality validation scheduled for Week 4 QA)*
- [ ] All sources tagged: country, region, time period (Data team leads; ml-eng consumes)

---

## 5. Cross-reference to existing `TASKS.md`

| `TASKS.md` item | Status | Sprint 1 mapping |
|---|---|---|
| §1 Web-augmentation guardrails + Tavily handling | ✅ done | Underpins the no-data fallback (§2.1); prompt copy still needs a rewrite |
| §2 Re-enable and tune the reranker | ⬜ | Week 3 (§2.5) |
| §3 Split Qdrant collections by document type | ⬜ | Week 1 → Week 3 (§2.4) |
| §4 Fix citation rendering + enrich source metadata | ⬜ | Week 3 (§2.6) |
| §5 RAG-side memory plumbing | ⬜ | Week 2 (§2.3) |

---

## 6. Open questions for Chidi

1. ~~**Target `RAG_GENERATE_TEMPERATURE`**~~ — **resolved: 0.7** (Jul 2 — team-agreed, favouring response variety; synthesis-on-gaps risk handled by code guardrails).
2. **ACF band thresholds + plain-language copy** — do we already have canonical wording, or do we draft and get sign-off? (Week-2 blocker for "ACF visible by default".)
3. **5 stakeholder groups + "major geographies"** — which exact list for the Week 4 QA sweep?
4. **User-profile payload shape** — what fields does backend send us for the session/profile context injection?
5. **Namespace boundaries** — is the 3-collection split (`policies`, `news_public_reports`, `academic_papers`) final, or should we also carve out a fourth for structured/insight-oriented sources (see `TASKS.md` post-launch item)?
6. **QA sign-off doc format** — is there a template, or free-form brief?

---

## 7. Out of scope for ml-eng (flag to Chidi)

- **4 UI bugs**: credit usage visibility, category count on free tier, mid-conversation category switching, query persistence during generation. → Frontend team.
- **Data intake doubling + geographic tagging at source** → Data team (`data-eng/`).
- **Data gap mapping, reconstruction, predictive scoping** → Research.
- **Beta list, partnerships, funding opportunity research** → Business.

Ml-eng consumes the tagged data and provides the intelligence layer QA sign-off; those items are dependencies, not our deliverables.

---

## 8. Working rules for this sprint (from slide 10)

- **Full focus.** Sprint 1 is priority over everything except genuinely urgent external commitments.
- **No late surprises.** If it is visible by July 14 that something won't ship by July 31, escalate then — not on July 30.
- **Quality over volume.** Doubling data intake only matters if quality and geographic tagging is right.
- **Brief format.** Anything requiring Chidi's review comes as a short document — not a Slack message, not a verbal update.
- **Test-2 bar.** Every critical and high issue from test 1 must be resolved before August 1. No exceptions.

---

## 9. Appendix — verbatim slide text

<details>
<summary>Click to expand slide-by-slide transcript</summary>

### Slide 1
```
OPENTRACE AFRICA
Sprint 1
Implementation Plan
July 1 – July 31, 2026
What we are building. What we are fixing.
What every department delivers by July 31.
opentrace.africa
```

### Slide 2 — WHY THIS SPRINT EXISTS
The internal test told us what to fix. This sprint is how we fix it.

Nine team members tested Ask ADZA across 72 hours in late June. The platform held technically, zero crashes, no hallucinations on specific queries, strong UI scores. But the intelligence layer showed clear problems that must be resolved before any external user sees the product.

- **Data routing broken** — RAG vector database unsegregated. African agricultural queries pulling from unrelated global academic content.
- **LLM fills gaps badly** — When ADZA has no data it synthesises academic prose instead of returning a clean, structured gap acknowledgment.
- **ACF signals invisible** — 5 of 9 testers did not notice the confidence signals at all. The core differentiator is hidden from users.
- **4 UI bugs identified** — Credit usage visibility, category count on free tier, mid-conversation category switching, query persistence during generation.

Sprint 1 runs July 1 to July 31. Internal test 2 begins August 1. Every critical and high issue from test 1 must be resolved before then.

### Slide 3 — THE BIGGER PICTURE
From Sprint 1 to commercialisation:
- Sprint 1 (Jul 1–31) — Fix cycle 1, all depts ← **We are here**
- Test 2 (Aug 1–3) — Internal test, all tiers
- Updates (Aug 4–5) — Rapid fixes post test 2
- Sprint 2 (Aug 6–Sep 6) — Fix cycle 2, final prep
- Private Beta (Sep 6–20) — First external users
- Post-Beta (from Sep 20) — Polish & commercialise

Sprint 1 week-by-week:
- Week 1 (Jul 1–7): Critical fixes: data routing + intelligence layer
- Week 2 (Jul 8–14): Data quality, ACF visibility, UI bugs
- Week 3 (Jul 15–21): Integration, response quality, data depth
- Week 4 (Jul 22–31): QA, stabilisation, sprint 1 sign-off

### Slide 4 — WEEK 1 (Jul 1–7) — Critical fixes: data routing and intelligence layer

**Data**
- Begin doubling the data intake pipeline across all priority source categories
- Audit all current data sources, identify ingestion gaps by geography and crop type
- Implement initial namespace separation in the RAG vector database

**Intelligence**
- Lower LLM temperature, ADZA must stop synthesising when it has no data
- Rewrite the no-data fallback prompt to produce a clean structured gap response
- Begin source purity enforcement on geographic and temporal query metadata

**Research**
- Begin data gap mapping across topics, geographies, crops, and time periods
- Identify highest-priority gaps for data reconstruction work in weeks 2 and 3

**Business, Partnerships & Fundraising**
- Begin compiling private beta waitlist, target 50+ qualified contacts by end of week
- Identify 4 new potential partnership targets not already in the current pipeline

**End of week 1 check:** Has LLM temperature been adjusted? Is namespace separation in place? Is the beta list started?

### Slide 5 — WEEK 2 (Jul 8–14) — Data quality, ACF visibility and UI fixes

**Data**
- Continue data intake acceleration, doubling volume on track
- Source quality audit: every ingested source tagged with country, region, and time period
- Geographic metadata tagging mandatory before any source enters the retrieval pool

**Intelligence**
- Surface ACF confidence signals on every response by default, band, score, plain-language note
- Implement session context isolation: no information bleeding between queries in a session
- Run structured test queries to validate the fallback prompt rewrite from week 1

**Research**
- Begin active data reconstruction work on highest-priority gaps from week 1 mapping
- Predictive intelligence scoping: identify which agricultural signals have sufficient historical depth

**Business, Partnerships & Fundraising**
- Private beta list at 75 or more contacts by end of week
- Initial outreach to all 4 new partnership targets underway
- Begin researching the 3 catalytic funding opportunities

**End of week 2 check:** Are ACF signals visible by default? Is source tagging in place? Is the beta list at 75+?

### Slide 6 — WEEK 3 (Jul 15–21) — Integration, response quality and data depth

**Data**
- Validate namespace separation with test queries, confirm no cross-contamination across geographies
- Data intake at minimum double pre-sprint volume by end of this week
- Begin building coverage for countries and crop types with the weakest current data

**Intelligence**
- Fix all 4 UI bugs: credit visibility, category count, mid-conversation switching, query persistence
- Full integration testing of the query → response pipeline end to end
- Response quality test: direct questions must return a direct answer before context and sources

**Research**
- Deliver first data reconstruction output for at least one high-priority gap area
- Submit interim research note to Chidi covering gap mapping findings so far

**Business, Partnerships & Fundraising**
- Private beta list finalised at 100 or more contacts, segmented by stakeholder type
- First substantive conversation completed with at least 2 of the 4 new partnership targets
- Funding opportunity research complete, one-page summary of each opportunity drafted

**End of week 3 check:** Are the 4 UI bugs fixed? Is the beta list at 100+? Has the research interim note been submitted?

### Slide 7 — WEEK 4 (Jul 22–31) — QA, stabilisation and sprint 1 sign-off

**Data**
- Full data audit completed: what is ingested, what is tagged, what tier, what gaps remain
- Data status brief ready for internal test 2 briefing on August 1

**Intelligence**
- Full pipeline QA: structured test queries across all 5 stakeholder groups and all major geographies
- Zero open critical issues by July 31, every issue from internal test 1 demonstrably resolved
- Pipeline QA sign-off document submitted to Chidi

**Research**
- Sprint 1 research brief submitted: gap mapping, reconstruction progress, predictive intelligence scoping
- Brief feeds directly into the sprint 2 research plan

**Business, Partnerships & Fundraising**
- 100+ beta contacts confirmed and segmented, list submitted to Chidi
- At least 4 new partnership meetings surfaced, confirmed or in active conversation
- 3 catalytic funding opportunity summaries submitted to Chidi for review
- All private beta outreach materials ready to deploy on August 1

**July 31 gate:** Zero open critical issues. All 4 departments have submitted their deliverables. Test 2 begins August 1.

### Slide 8 — WHAT EVERY DEPARTMENT DELIVERS BY JULY 31

- **Data:** Data intake 2× pre-sprint volume · All sources tagged by geography and time · RAG namespace separation live and validated · Full data audit document submitted
- **Intelligence:** LLM temperature lowered, no synthesis on gaps · Fallback prompt rewritten and tested · ACF signals visible by default on every response · Session context isolation live · 4 UI bugs fixed · Pipeline QA sign-off submitted
- **Research:** Data gap mapping completed · First reconstruction output delivered · Predictive intelligence scoping note submitted · Sprint 1 research brief submitted to Chidi
- **Business, Partnerships & Fundraising:** 100+ beta contacts, segmented by stakeholder type · 4 new partnership meetings surfaced · 3 funding opportunities with one-page summaries · Beta outreach materials ready to deploy

### Slide 9 — SPRINT 1 MASTER CHECKLIST · JULY 31 GATE

**Data:** Data intake volume doubled · All sources tagged: country/region/time period · RAG namespace separation implemented and validated · Data audit document completed and submitted

**Intelligence:** LLM temperature reduced, no academic synthesis on gaps · No-data fallback prompt rewritten and tested · Source purity enforced on geographic and temporal queries · ACF signals visible by default on every response · Session context isolation, no cross-query contamination · 4 UI bugs fixed and verified · Pipeline QA completed · Zero open critical issues by July 31

**Research:** Data gap mapping complete · First data reconstruction output delivered · Predictive intelligence scoping note submitted · Sprint 1 research brief submitted to Chidi

**Business, Partnerships & Fundraising:** 100+ beta contacts, segmented by stakeholder type · 4 new partnership meetings surfaced · 3 catalytic funding opportunity summaries submitted · Beta outreach materials ready to deploy

### Slide 10 — HOW WE WORK DURING SPRINT 1

- **Full focus.** This sprint is a priority over everything except genuinely urgent external commitments.
- **No late surprises.** If you can see by July 14 that something will not be done by July 31, say so then, not on July 30.
- **Quality over volume.** Doubling data intake only matters if the quality and geographic tagging is right. Bad data in makes the intelligence problem worse.
- **Brief format.** Anything requiring my review comes as a short document. Not a Slack message. Not a verbal update.
- **The test-2 bar.** Every critical and high issue from test 1 must be resolved before August 1. No exceptions.

> The UI held. The guardrails held. The intelligence layer is not ready. That is what we fix in July.

</details>
