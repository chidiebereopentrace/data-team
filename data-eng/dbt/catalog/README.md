# Generated dbt source catalog

`sources.generated.yml` is produced from the BigQuery schema catalog and is **committed**, so a clean clone (and CI `dbt parse`) works without BigQuery access.

Regenerate and commit it whenever BigQuery schemas change:

```bash
cd data-eng
python data/local/scripts/generate_dbt_sources.py --refresh
```

`dbt_project.yml` includes this directory in `model-paths` so dbt loads the generated sources alongside hand-written YAML under `models/` (`sources_ingestion.yml`, `sources_geoboundaries.yml`, …).

Do not hand-edit the generated file.
