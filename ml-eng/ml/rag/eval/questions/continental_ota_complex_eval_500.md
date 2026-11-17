# Continental OTA complex evaluation cases

The [JSONL dataset](continental_ota_complex_eval_500.jsonl) contains 500 additional Ask ADZA cases. It has 300 English cases and 200 French cases. The development split contains 120 English and 80 French cases. The held-out split contains 180 English and 120 French cases.

- **Policy** has 167 cases, including budget allocation, programme timelines, targeting, and rules for releasing reserve funds.
- **Business** has 167 cases, including market entry, capital allocation, supplier diligence, operational risk, and scenario analysis.
- **Research** has 166 cases, including cross-country harmonization, reproducible extracts, literature review, measurement bias, and causal identification.

Each initial question calls for two related indicators and an explicit analysis with evidence checks. The suite includes 84 cross-country cases and 125 follow-up turns. There are 85 structured, 161 document, and 254 mixed cases. The cases collectively cover all 24 Continental OTA categories, 214 sub-indicators, 238 value chains, 54 countries, and six Qdrant corpus types.

`baseline_case_id` identifies the scope anchor in the [original multilingual suite](continental_ota_eval_500.md). Cases inherit that anchor's `scenario_group` and split. Keep held-out cases in both sets out of prompt examples and tuning. The prompts use 18 task designs with different countries, indicators, value chains, and periods.

- Send `turns[].user` in order and retain the system's actual reply before each follow-up. The other fields are for evaluation.
- `expect` records the requested language, places, time, evidence types, and verification checks. A publication window refers to when a document was published. An observation window refers to when a measure was observed. Some cases ask for both observed history and a separate forecast.
- `catalogue_scope` records Continental OTA IDs and source snapshot coverage for each requested place. These IDs link cases to the dashboard catalogue. Use a verified Mart schema and metric binding when checking structured evidence. The dashboards and label meanings are described in the original suite's guide.
- Check every requested indicator and place. Evaluate whether comparison periods, units, and denominators match. A case with missing evidence can pass when the answer identifies the specific gap and qualifies its conclusion.

The catalogue coverage and country-to-value-chain pairings were checked against the local Continental OTA build dated November 8, 2026. The prompts use shared task designs and still need editorial and native French review. Exact Mart observations and Qdrant passages still need verification for gold scoring.
