{{ config(materialized='table') }}

-- Gold: forestry — source table fao_rl not yet ingested into BigQuery.
-- Registered with correct schema; populated once ingestion is complete.

select
    cast(null as string)                           as forestry_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as product_key,
    cast(null as string)                           as unit_key,
    cast(null as numeric)                          as production_quantity,
    cast(null as numeric)                          as trade_value,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy)
where false
