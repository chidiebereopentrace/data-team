{{ config(materialized='view', enabled=false) }}

-- Unmapped: no raw_dev tables in staging_dev_taxonomy.yml yet.
-- Set `tables:` in the taxonomy and re-run the generator (this file is not overwritten).
select
    cast(null as string) as _unmapped
