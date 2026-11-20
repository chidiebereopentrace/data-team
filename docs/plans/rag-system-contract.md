# Complete the shared RAG evidence contract

Status: proposed implementation sequence, based on the current checkout.

## Objective

Every requested part of an answer must have verified evidence or an explicit, accurate gap. The same request must govern planning, retrieval, calculations, evidence packing and generation across OpenTrace's domains and supported languages.

Keep OpenRouter for local and deployed inference. Exercise local QA through the authenticated HTTP API. Keep the deferred Qdrant rebuild on its separate branch.

## What exists, and what remains incomplete

- `TurnContract` preserves the original question, language, facts and document questions. `FactRequirement` now carries a clause, requirement ID, geography, entities and time windows. Explicit member selection, discovery scope, requested dimensions, operations and presentation requirements still need to be carried consistently; document requirements still lack independent scope.
- `evidence_requirements.py` plans and checks explicit country-level, whole-year comparisons. It is not yet a common planner and coverage model for all operations and grains.
- `routing_plan.py` records source decisions, but also interprets raw question text, decomposition and task mode independently of the contract.
- `graph.py` still switches behavior on `plan_source == "turn_requirements"` at several stages. Other request paths have different guarantees.
- `typed_pack.py` still has an explicit `price_report` exemption from top-k truncation. Required evidence needs a shared retention rule.
- Verified numerical rendering currently has separate price-report and annual-comparison paths. Their correctness guarantees need to become shared across supported observations and operations.
- Request deadlines and bounded worker pools already exist. Their behavior needs end-to-end verification under partial failures and concurrency.

Reuse these modules. Replace their competing decisions and obsolete paths as each supported operation passes verification; do not add another planner beside them.

## 1. Make requirements explicit and independently scoped

Extend the existing contract with stable requirement identifiers and the scope of each fact or document requirement. A fact needs its measure, requested members, geography, period, grain, operation and relevant dimensions. A document requirement needs its question and relevant scope. Units and comparability constraints must be established against the actual catalog and returned observations.

Preserve the original question and its explicit constraints. Language enrichment can provide canonical terms; it cannot silently replace a requested crop, indicator, country or period. Keep explicit requests distinct from supporting evidence inferred for an explanation.

Keep useful question expansion enabled. Suggested crops, related indicators, explanatory questions and additional searches should improve discovery. Record which requirements came from the user and which were inferred, so an expanded example list cannot silently become an exhaustive product filter, change the requested grain or replace an explicit place or period. This applies to English as well as translated and code-mixed questions.

Use one slot validator. A broad report may have a valid discovery scope without naming one commodity. Clarify only when useful retrieval genuinely cannot proceed. Do not turn a broad request into an unstated complete basket or a fabricated indicator.

Pass when differently scoped clauses survive serialization and compilation independently, including a question asking for Kenyan yields in two years and Ugandan employment over a different period. Mutating a decomposition's geography or language must not erase explicit question constraints.

## 2. Compile one execution plan against the real catalog

Compile every requirement into supported warehouse work, document searches, or an explicit unsupported status. Use actual Mart tables, columns, discriminator values, dimensions and units. Evaluation labels are not warehouse column names.

Reuse the existing bind compiler and class engines behind this interface. Allow multiple measures to share a table and query when their scopes and outputs remain distinguishable. SQL count is diagnostic; requirement coverage is the acceptance criterion.

Mixed analytical requests normally use both BigQuery and Qdrant. Structured facts can use Qdrant for relevant indicator context and reported figures. Document questions retain their document requirements even when a warehouse measure is absent. Record every selected or skipped source and its reason.

Choose eligible corpora from the evidence question and corpus contents, including news, public reports, policies, research and OTA where relevant. A single indicator class must not silently exclude a relevant evidence source. An explicit data-only request can omit narrative retrieval with a recorded reason.

Pass when each requested indicator has supported work or a specific unsupported reason. No named requirement may disappear because a different indicator won a priority score.

## 3. Return typed evidence and per-requirement outcomes

Associate observations and passages with requirement IDs. Warehouse evidence carries the actual measure, members, place, observation period, grain, unit, currency where relevant, source and SQL provenance. Document evidence carries a passage, source identity, scope and the requirement it may support.

Use consistent outcomes: satisfied, partial, unsupported, no match, rejected, timed out, unavailable and intentionally skipped. Distinguish an empty successful search from a failed search. Preserve partial hits when a corpus times out after returning evidence.

Give distinct observations distinct identities. A shared publisher, URL or source identifier does not make different countries, periods or measures duplicates. A passage does not satisfy a requirement merely by mentioning its terms.

Pass when one failed requirement leaves other completed evidence intact, and source metadata cannot collapse distinct observations.

## 4. Make packing preserve required coverage

Apply document relevance ranking to passages and retain the bounded structured observations needed for the requested operations. Use SQL aggregation and comparable series summaries for large reports; do not send unbounded raw warehouse results to a model.

Check coverage before and after packing. If a text budget cannot represent a requested part, report that limit explicitly. Protect every requested country, period and indicator ahead of redundant evidence. Document requirements must also retain supporting passages when available.

Replace the `price_report` exemption with this shared rule. Keep one final owner for model-context budgets instead of introducing additional destructive cutoffs during retrieval.

Pass when the same small-budget test preserves required coverage for prices, employment, production, climate, trade and food security, including relevant documents.

## 5. Separate verified calculations from narrative synthesis

Compute supported differences, percentage changes, percentage-point changes, rankings and aggregates from compatible typed observations. Generic operations must respect domain semantics: quantities, rates, stocks, flows, indices and yields cannot be treated as interchangeable values.

Render verified findings and evidence gaps through a shared answer representation. Keep source-specific semantic bindings where needed. Replace duplicated report assembly and price-only guarantees as the common operations become supported.

Use the LLM to explain and organize evidence in the requested language. Explanatory claims need passages that support them; numerical co-movement does not establish causation. Interrupted, uncited or unsupported model prose must not displace the verified findings or ask users to narrow an already valid request.

Pass when a model timeout, length limit or claim that a verified series is missing leaves a complete, readable account of the supported findings and precise remaining gaps across domains.

## 6. Enforce the request deadline across the complete operation

Use the existing request budget as the common deadline. Start independent retrieval together, reserve time for synthesis and rendering, and propagate remaining time to actual backend calls. Bound concurrency, retry only a specific recoverable requirement, and prevent work from accumulating after client deadlines.

Measure decomposition, embedding, each corpus search, SQL preparation/execution, reranking, generation and rendering. Distinguish cold and warm latency. Set release thresholds from measured results; do not promise a latency target before measuring it with the intended provider and concurrency.

Pass when a stalled backend cannot erase completed work or monopolize workers, including concurrent authenticated API requests.

## 7. Make cross-domain API evaluation the release gate

Use the existing multilingual and complex Continental OTA question sets as coverage inventories. Review expected requirements and map them to actual catalog capabilities; generated questions alone do not establish correct answers.

Create a small reviewed release set across production/yields, inputs, climate/soils, food security, trade, employment/livelihoods and policy/research. Include broad reports, mixed questions, distinct per-clause periods, regional/continental scopes and missing data. Cover English, French and the configured African languages, with native review for meaning and answer quality.

Deterministic CI checks must use the same contract, execution and answer interfaces as callers. Authenticated API checks must exercise real BigQuery, Qdrant and OpenRouter separately from controlled dependency-failure tests. Track requirement completeness, citation support, unsupported claims, silent evidence loss and latency by domain and language. A high total score cannot conceal a failed domain or language slice.

Keep the QA client and API on shared persistent local credentials, with invalid requests still rejected. Record safe build/configuration identifiers so an old running process is distinguishable from the code being tested.

## First implementation slice

Start with independently scoped requirements and their shared compiler/validator. Take a mixed two-country, two-indicator question through the real API, alongside a document question and a broad report from another domain. Add a multilingual variant and a failed-backend run.

For each case, assert the same requirement IDs and scope in the compiled work, returned evidence, packed evidence and final answer or explicit gaps. Extend that working path to other operations and grains, deleting the obsolete route in the same change. Keep each increment working end to end.

The system-level issue is resolved only when these guarantees hold across the supported domains and languages, including failure cases. Successful price reports and a large unit-test count are insufficient by themselves.

## Implementation checkpoint — 19 November 2026

The country-level annual fact path now carries a clause, stable requirement ID, countries, entities and time windows for each requirement. Source selection, binding, SQL validation, row enrichment, coverage and generation use that local scope. Bindings are keyed by requirement rather than table, so production and yield, or two separately scoped yield requests, can use the same serving table without overwriting each other. Valid sibling bindings survive a capability miss for the primary measure.

Bound observations retain their requirement IDs through retrieval and packing. Passage top-k limits do not truncate their already bounded row sets. Coverage distinguishes queries that did not run, failed validation, timed out, completed without matching observations, or lacked a verified binding. English annual fact answers can render cited observations and compatible endpoint changes independently of explanatory generation; incomplete generation no longer dumps raw source metadata.

The captured Kenya-policy lookup also checks document-title identity and edition before accepting a passage. A mention of the requested plan in an unrelated report does not establish what that plan says. Empty or incomplete document searches do not establish that the policy omits a topic. Country recognition now retains accented names with typographic apostrophes, including Côte d’Ivoire, and indicator matching treats word-separating hyphens consistently.

A deterministic recovery test exposed repeated Redis connection probes during an outage. A process now waits 30 seconds before retrying a failed connection, while still checking for recovery. This does not establish Redis availability or shared-session durability.

This checkpoint does not close the complete production-readiness plan. Price-series compilation remains distinct because of market, package and currency grain. Seasonal, subnational and regional requirements, exact climate and food-security indicator definitions, agricultural labour productivity and sex-specific employment denominators still need catalog-backed acceptance cases. Native-reviewed language coverage and the deferred Qdrant rebuild remain separate release requirements. Do not infer that the seven captured analytical reports all pass from the annual-country regressions.

Validation at this checkpoint: 476 targeted retrieval/generation/session tests passed, with one optional fakeredis test skipped; 64 API, UI and date-grounding checks passed separately. The authenticated local API replay executed two successful warehouse queries, retained all 21 observations, and cited all 21 for the independently scoped yield/employment question. Ghana's 2015–2023 observations survived, with 2024 explicitly unverified. The named-policy replay returned no verified passage and made no claim about the plan's contents.

The two live requests took 45.97 seconds and 14.10 seconds respectively. These are individual observations, not a p95 measurement or an established improvement. The mixed request spent 17.31 seconds in OpenRouter decomposition and 12.88 seconds in generation, which timed out; verified figures survived that timeout. Existing Qdrant collections still reported incompatible ingest versions. Latency and live documentary evidence remain open release conditions.

### Generation and language follow-up

Analytical generation now allows 8,192 tokens at temperature 0.6, with a bounded 120-second request deadline. The former 256-token report clamp and 120-word price-commentary instruction have been removed. Short lookup and export-caption budgets remain bounded separately. Local QA settings and the deployment environment template use the same values; no production deployment was performed.

Finish reasons are tracked by LLM purpose. A capped decomposition or reranking call no longer marks the final answer as truncated. When incomplete model commentary is replaced by a complete verified table and a coverage note, that replacement does not inherit a false truncation warning.

The repetition guard now looks for dominant consecutive loops, excluding analytical tables and citations. Reusing an indicator name across countries does not establish degeneration. Genuine generation failures no longer claim the requested language is unsupported, and French failure/length messages remain in French. The HTTP inspector now carries the API's answer language and generation-limit fields.

The supplied French question returned a 4,815-character French response with 15 citations and no generation length cap in 69.48 seconds after the deadline adjustment. The employment question returned its verified table with 27 citations and no false length warning in 10.08 seconds. These runs verify the settings and removal of the language refusal; they do not establish that the retrieved indicators or all resulting claims are correct. The French run exposed production/area wording confusion, addressed with explicit dimension instructions, and still lacked some requested series. Retrieval bindings and multilingual claim verification remain release requirements.

## Confirmed regression — itemized production discovery, 20 November 2026

Status: diagnosed and reproduced; application fix and live acceptance remain open. Add this case to the first implementation slice before widening the rollout. Keep the expansion capability: the user explicitly wants richer interpretation and discovery.

### Captured behavior and reproduced cause

Question: "give me a table itemizing the crop production in nigeria in 2020".

The English enrichment proposed ten crops, metric-ton quantities and regional detail. BigQuery was selected for production, but the requirement planner returned `skip_bq=true`, no tables or bindings, and `max_sql_queries=0`. No SQL was submitted. The compiler currently skips a crop-dependent requirement unless its crop regex recognizes exactly one crop. Both a broad inventory with no named crops and a request with several named crops therefore disappear from executable warehouse work. The accepted empty plan then ends planning in `graph._build_bq_sql_plan`.

The warehouse catalog describes `agg_production_country_year` at country × product × year × source grain, with `production_qty` in tonnes, `area_harvested` in hectares and `yield_recomputed` in tonnes per hectare. `dim_product` supplies product identity. This establishes a supported serving structure, not that every requested Nigeria/2020 observation exists; that must be checked by executing the scoped query.

Qdrant returned 18 public-report and 10 OTA candidates. Six passages reached generation. The answer did not produce the requested table, discussed COVID cases and regional trade instead, and cited one report without supporting the requested crop quantities. Fact coverage said `unsupported`, while the generation plan still said `served`, `evidence_tier=strong` and `must_ground_in=any`, despite having no usable numerical evidence. Those states must not imply the numeric request was fulfilled. Absence from retrieved context cannot establish absence from the warehouse when its query never ran.

An offline replay of the actual contract and requirement compiler failed for the captured expansion, the original broad inventory, and an explicit maize-and-rice request. A single-maize control compiled one valid binding. Replay command from the repository root, with `PYTHONPATH=ml-eng` and UTF-8 enabled:

```powershell
C:/Users/chidi/anaconda3/python.exe ml-eng/data/local/.rag-qa/repro_crop_inventory_nov20.py
```

The ignored local replay uses the supplied capture and makes no backend calls. Its current result is three failing cases and one passing control. Promote these cases to durable, self-contained regression fixtures during implementation; this diagnostic script is not a release gate.

### Required implementation, integrated with steps 1–5

1. **Keep expansion, preserve authority.** Carry the original request, validated canonical meaning and inferred suggestions separately within the existing contract. Here, country=Nigeria, observation year=2020, measure=production, product breakdown and table presentation are user requirements. The suggested crop examples and regional exploration are discovery aids. They may launch additional relevant work but cannot restrict the answer to an invented fixed basket, force regional grain onto national observations, or overwrite explicit filters. Ground translated aliases semantically; do not require valid translations to reproduce the original words literally. Expose accepted expansion and its role in the inspector.

2. **Compile explicit members and broad member discovery.** Replace the exactly-one-crop condition with catalog-backed handling of a named member set, a supported group, or all available relevant members within the requested scope. For this request, compile bounded country/product/year/source work against the production aggregate and product dimension, selecting `production_qty` explicitly. Resolve labels against the actual dimension, including multiword names; a regex's failure to recognize an expanded term must not erase valid sibling work. Do not assume the ten suggestions are all crops, all supported, or the complete answer. Reuse the existing bind/compiler path; do not introduce a crop-specific alternate planner or model-written-SQL escape.

3. **Retain member identity end to end.** Observation identity, coverage, deduplication, packing and verified rendering must include the relevant product and metric dimensions, alongside country, period, source and unit. The current annual observation key is requirement × country × year and cannot safely represent several crops under one requirement. Changing only the compiler would risk treating different crop quantities as contradictory values. Apply the same rule to market/price type, employment breakdowns and other supported domain dimensions. Do not combine incompatible sources or count product aggregates together with their constituent crops.

4. **Honor itemized presentation and complete bounded coverage.** Preserve the requested product breakdown and table output independently of the task-mode label. Render product, production quantity, unit, year, country and source from verified observations. Retain every supported member within the declared result bound; use deterministic summaries or an explicit continuation/export when necessary. Neither a ten-row cap nor passage top-k may silently turn a partial inventory into a complete list. State what was queried, returned, excluded or unresolved. A discovered list is the available catalog coverage, not a claim to include every crop grown in the country.

5. **Require evidence appropriate to the claim.** Reports and OTA can supply figures as well as explanation, but a numeric claim needs a matching product, place, observation period, quantity, unit and attributable source. Corpus type, country mentions or a strong source tier alone do not satisfy production coverage. Reconcile conflicting series or report them separately. An unrelated COVID or trade passage must not support a production figure or a claim that production data is absent. With no executed BQ query, expose the planning failure accurately; do not present it as a successful empty warehouse result. Generation status and confidence must reflect satisfied requirements and supporting claims, not just retrieved passage counts.

### Acceptance and release checks

- Replay the exact broad question, the captured expanded form, an explicit two-crop request, a supported crop-group request, and the single-crop control. All catalog-supported shapes must create executable scoped work; unsupported members retain specific reasons without blocking valid members.
- Include a non-production inventory from another domain to prove this is a shared member-selection and breakdown capability. Test an expansion that proposes unsupported members and regional exploration: it must preserve the user's place, year and national table requirement while keeping useful additional searches.
- Use fixtures with more than ten products and deliberately distinct area, production and yield values. Verify the production column and tonnes, exact source attribution, product identity after serialization, completeness after packing and the requested table shape. Distinct products must survive even when publisher, country and year are identical.
- Test sparse products, null quantities, conflicting sources, overlapping aggregate/member products, unresolved dimension labels and a bounded result that needs continuation. Never convert missing observations to zero or manufacture an exhaustive basket or total.
- Supply unrelated COVID/trade passages with high retrieval scores, and separately a relevant report/OTA passage with a verifiable crop figure. The first cannot satisfy numeric coverage or earn a supporting citation; the second may support its scoped claim without being passed off as a BQ result.
- Exercise successful empty queries, compiler rejection, BQ timeout and unavailable service as distinct outcomes. Preserve valid documentary figures and valid sibling warehouse work. An unexecuted query cannot produce an asserted warehouse data gap.
- Run authenticated HTTP API acceptance with BigQuery, Qdrant and OpenRouter, using the same local/deployed configuration. Check planned and submitted SQL, member coverage before and after packing, final table values and units, citations, and stage latency. Inspect the real Nigeria/2020 result before claiming the data is present or absent. The separate Qdrant rebuild is not part of this application fix.

## Implementation update — inventories and multilingual subject preservation

The inventory implementation now compiles bounded explicit product sets and catalog discovery through the existing requirement/bind path. It selects production quantity explicitly, retains product identity in annual coverage, protects the original inventory's place, period and requested breakdown from added suggestions, and renders itemized observations with their sources. Crop discovery uses `dim_product.cpcv2`, distinguishing primary crops from animal and processed products; cereal discovery uses the catalog group rather than a hand-picked basket. The classification is grounded in the [UNSD CPC v2 hierarchy](https://unstats.un.org/unsd/classifications/Econ/Structure/Detail/EN/1073/01). Unclassified products are excluded and the answer states that limitation.

Two additional loss points were reproduced and changed. Source-header allocation now includes requirement annotations, so adding a figure-candidate label cannot cut the figure out of a short passage. Verified numerical tables and their citation registry use the complete SQL-bounded observations independently of the model's prose context budget. Regressions cover 110 product observations, a four-passage packing limit, missing product siblings, unexecuted queries, multiword labels, and a relevant reported quantity when BQ times out. An unrelated COVID passage cannot establish a production figure or a warehouse data gap.

The 118-row authenticated run is not acceptance: it included animal and processed products and replaced verified rows with an uncited summary. The CPC filter and the numerical citation registry were added after that run.

Authenticated acceptance, 2026-11-20, through the local inspector at `http://127.0.0.1:8503/`, which posts to `POST /query/integrated` on `http://127.0.0.1:7860`. The API process that had been listening since 19 November compiled the broad crop question as six suggested product names plus a subnational geography filter, BigQuery returned zero rows, and the answer said production could not be verified. That process was restarted from this checkout. The same question then compiled one bind-compiler query against `agg_production_country_year`: `production_qty`, Nigeria, year 2020, and `dim_product.cpcv2` normalized to the CPC prefix `01`. The public trace redacts SQL literals; the statement contains `cpcv2` and does not contain `product_name IN` or a `geo_level` filter. There was no separate planned-SQL list. BigQuery returned 52 rows (job status `ok`). A direct count of Nigeria 2020 production rows with a value found 106 rows: 52 with that CPC prefix and 54 outside it, including meat, milk, eggs, hides, oils, beer, sugar, tea, and FAO aggregate totals. Those 54 are not in the answer. Packing kept all 52 warehouse rows (candidate context 80, structured 52, packed 52). The rendered table uses tonnes (`t`) and one `structured_data` citation per product, `FAOSTAT — Production qty — {product}`. The answer is that table plus the catalog-coverage note. No model summary replaced the rows. The inspector displayed the same 52-row table.

### Added capture: Igbo question about agricultural soil/farmland

Original question: "kedu ihe iwere ikwu banyere onudo aja alaugbo di na kenya". The known soil/farmland terms in the original must survive interpretation; the complete intended meaning of informal spelling still needs native review. The captured planning question instead became "What were the factors influencing the adoption of improved maize varieties in Kenya?" The run retrieved 100 BQ rows and 34 document candidates, packed 18 items, cited none, and returned an English generation-failure message while `answer_lang=ig`. Its generation ceiling was 512 tokens. These observations establish subject drift and unsuitable response handling; they do not establish the exact upstream generation stop reason.

The HTTP trace previously omitted enrichment and query-view fields. Therefore its displayed `language_enrich=NULL` and `language_enrich_applied=false` did not prove that enrichment was disabled. Trace serialization and inspector normalization now retain the actual enrichment, source, subject-guard status and query views. Generation traces additionally expose the last generation finish reason; deadline exhaustion, busy HTTP workers, timeout and request error are recorded separately from token-length truncation.

Implemented controls for this failure:

- Preserve useful expansion while checking known original subject terminology. An expansion retaining soil and adding related yield context remains valid. A candidate replacing the soil subject with maize-variety adoption is rejected as the canonical request; planning retains the full original wording with a known bilingual subject anchor. The same protection applies if enrichment returns no text, and its provenance survives caching.
- Do not persist the rejected candidate's crop, indicator list or explanation topic into the active request. Generic indicator words such as "soil" must not become a saved commodity either.
- Route declared diagnosis/report/synthesis jobs to the analytical generation budget, while preserving explicit data-only/export precedence. A broad assessment must not inherit the single-fact 512-token ceiling merely because it names one country.
- Keep an Igbo generation-failure response in Igbo. A model failure is not evidence that the language, requested subject or data is unsupported.

These are bounded safeguards, not a claim of complete multilingual semantic verification. The initial soil anchors cover English, French, Igbo and Swahili; native-reviewed subject and alias coverage across all 17 configured languages remains required. The captured question, clean spelling, code-mixed variants, faithful expansions, incorrect expansions and failed/cached translations must be included in the reviewed release set.

Remaining soil-domain acceptance must inspect the actual catalog and results: retrieve the relevant available soil properties rather than silently selecting CEC as a proxy for all soil health; retain property, depth, unit, source, administrative coverage and sample counts; distinguish static soil estimates and load dates from time-series observations; and do not treat a few points or an unweighted mean of administrative averages as a verified national assessment. Require supporting passages for interpretation. This belongs in the shared indicator/dimension discovery work, not a special-case translation or unrestricted scan of the point-level soil fact.

Verification status: inventory member-discovery tests, the productivity and employment citation tests, and the Research route, tenant, and GeoJSON tests passed on this checkout. Warehouse citations for an itemized production member now keep the product, so identical publisher and indicator rows do not collapse; a price report still groups to institution and indicator. Context allocation measures the source header that is actually written, so a later source id cannot clip the last character of a study that fits in the budget.

This does not close the evidence contract. Seasonal, subnational, and regional scopes, price grain, soil-property discovery, native review of the configured languages, the Qdrant rebuild, latency thresholds, and the previously reproduced full-suite failures remain. `news_data` and `public_reports` are still unhealthy against ingest version `2026.08-acf-claim-v2`. No production deployment was performed.
