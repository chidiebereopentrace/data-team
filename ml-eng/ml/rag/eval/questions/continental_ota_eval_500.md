# Continental OTA evaluation dataset

The [JSONL dataset](continental_ota_eval_500.jsonl) contains 500 scenarios for Ask ADZA. It has 200 development cases and 300 held-out cases. Exact scenario groups stay in one split. Keep the held-out cases out of prompt examples and tuning.

The [complex companion set](continental_ota_complex_eval_500.md) adds 500 English and French scenarios for policy, business and research decisions.

- **Languages.** All 17 named languages in [the answer language profile](../../chatbot/answer_language.py) have 29 or 30 scenarios. There are 29 conversations with a follow-up. Of those, 16 include a deliberate code switch.
- **Continental coverage.** There is a dedicated case for each of the 214 sub-indicators, including the 134 workbook entries and 80 dashboard additions. A separate case targets each of the 238 crop, livestock, and fisheries value chains across all 12 groups. All 24 indicator categories, all 54 dashboard countries, and all six Qdrant corpus types appear in the suite. A value chain is paired with a country that lists it in the dashboard.
- **Other scenarios.** The remaining 48 cases check follow-up corrections, observation versus forecast, or questions that require two indicators and document evidence. Across the full file, there are 299 structured, 123 mixed, and 78 document cases.

## Source and labels

The source is the [Continental OTA Dashboard](<../../../../../data-eng/Continental OTA Dashboard Codebase/README.md>). The workbook's **Indicators** and **Additional Indicators** tabs feed the dashboard alongside the additions and country-specific value chain assignments. Dataset scope was matched to the local dashboard build dated **2026-11-08**, SHA-256 `fa74303231f8193580ccb8b2ae306d5c60fea4bb219e61927714cb3ac4ac1368`. The dashboard coverage crosswalk provides candidate source and Mart table pointers for 57 distinct sub-indicators. A crosswalk entry can point at landing data without a serving Mart table.

- `turns[].user` is the prompt. Send each turn in order and retain the system's actual reply between turns. The prompt and conversation are the inputs. The expectation labels belong to the evaluator.
- `turns[].expect` records the country, value chain when named, time, answer language, and the evidence the question asks for. `requested_evidence.mart_facts` and `requested_evidence.documents` describe the task. A true flag can still lead to an evidence gap. `preferred_corpus` is the document type requested in the prompt.
- `turns[].catalogue_scope.indicator_ids` contains Continental OTA catalogue IDs used to check benchmark coverage. Those IDs identify what the question is about. Use the reviewed warehouse schema and metric mapping to bind any SQL query. A `null` `warehouse_crosswalk` in `dashboard_targets` means the dashboard has no source mapping for that catalogue entry.
- `dashboard_targets` provides the catalogue names, unit, frequency, and grain hints. `catalogue_scope.snapshot_status_by_indicator` records the dashboard status for the country and value chain named in the turn. The dashboard uses `F` for Mart coverage, `P` for landing-only coverage, and `G` for a gap in that snapshot. A `null` value means the dashboard gives no value-chain-level status for that indicator. Verify the exact metric, grain and period in the serving system before treating a case as answerable from the Mart.
- `checks` describes the answer behavior. A case passes only if it accounts for the requested evidence, scopes and follow-up. A properly explained evidence gap can pass a gap case.

The prompts and scope labels are generated drafts. Native speakers should review fluency, terminology, and code-switching. To promote a case to a gold label, confirm the exact Mart series and periods or record the relevant Qdrant passage IDs and the claims they support. Confirm known gaps against the same serving snapshot.

From `ml-eng`, inventory both suites without external calls:

```sh
python -m ml.rag.eval.run_ota_eval --inventory --split all
```

Replay four development cases against the current graph with local backend configuration:

```sh
python -m ml.rag.eval.run_ota_eval --case ota-complex-0116 --case ota-complex-0023 --case ota-complex-0049 --case ota-0007
```

The replay sends the original user text and actual previous replies with the fixture date. Its report records candidate counts, retrieval status by backend, citations, and elapsed time. Missing requested evidence can reflect a genuine coverage gap or a retrieval problem. Candidate counts can arrive without supporting the requested indicator, place, or period. Treat the report as a diagnostic baseline. Gold scoring requires reviewed claims and evidence for each requirement. The separate `run_answer_eval.py` command continues to build its own questions.

