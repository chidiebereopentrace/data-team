{{ config(materialized='table') }}

-- Gold: food balance sheet — source tables fao_tcl and fao_rfn not yet ingested into BigQuery.
-- Registered with correct schema; populated once ingestion is complete.

select
    cast(null as string)                           as food_balance_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as product_key,
    cast(null as string)                           as unit_key,
    cast(null as numeric)                          as value,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy)
where false
