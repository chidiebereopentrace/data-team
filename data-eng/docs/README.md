# data-eng documentation

| Doc | Audience / role |
|-----|-----------------|
| [OpenTrace_Mart_Complete_Guide.md](./OpenTrace_Mart_Complete_Guide.md) | Analysts — ERD + indicator-class writing (EL, GYI, PROD, FS, …) |
| [OpenTrace_Mart_Complete_Guide.pdf](./OpenTrace_Mart_Complete_Guide.pdf) | Same Complete Guide as PDF (Mermaid ERDs rendered) |
| [OpenTrace_Mart_Entity_Dictionary.xlsx](./OpenTrace_Mart_Entity_Dictionary.xlsx) | Analysts — branded catalogue, class map, insight template, **full Columns + descriptions** |
| [mart_dev_entity_dictionary.xlsx](./mart_dev_entity_dictionary.xlsx) | Engineers — Entities / Columns / Relationships / Recipes / ACF dump |
| [MART_DEV_OTA_ANALYST_GUIDE.docx](./MART_DEV_OTA_ANALYST_GUIDE.docx) | OTA insights analysts — report-writing playbook (Word) |
| [MART_DEV_OTA_ANALYST_GUIDE.md](./MART_DEV_OTA_ANALYST_GUIDE.md) | Same OTA guide in Markdown |
| [CATALOG_TO_MART_MAP.md](./CATALOG_TO_MART_MAP.md) | Catalog ↔ mart glossary, grain rules, ACF contract |
| [MART_QA_NOTES.md](./MART_QA_NOTES.md) | Mart QA inventory, caveats, rebuild notes |
| [GLOBAL_ARCHITECTURE.md](./GLOBAL_ARCHITECTURE.md) | End-to-end data platform architecture |

**Which mart doc?** OpenTrace Complete Guide + Entity Dictionary = indicator-class framing, ERDs, and full per-table Columns (same descriptions as the engineer dump). `mart_dev_entity_dictionary.xlsx` = Relationships / Recipes / ACF plus the same Columns sheet. OTA guide = how to write OTA reports against the warehouse.

Regenerate from `data-eng/`:

```powershell
python scripts/build_mart_entity_dictionary.py
python scripts/build_opentrace_mart_entity_dictionary.py
python scripts/build_opentrace_mart_complete_guide_pdf.py
python scripts/build_mart_ota_analyst_guide_docx.py
```
