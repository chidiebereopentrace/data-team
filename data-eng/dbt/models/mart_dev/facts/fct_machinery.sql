{{ config(materialized='table') }}

-- Gold: machinery — source table fao_ti not yet ingested into BigQuery.
-- Registered with correct schema; populated once ingestion is complete.

select
    cast(null as string)                           as machinery_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as technology_key,
    cast(null as string)                           as unit_key,
    cast(null as numeric)                          as machinery_stock,
    cast(null as numeric)                          as density,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy)
where false
