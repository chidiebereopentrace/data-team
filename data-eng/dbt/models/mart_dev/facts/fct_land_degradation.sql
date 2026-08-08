{{ config(materialized='table') }}

-- Gold: land degradation fact — schema registered; populated once a dedicated
-- degradation source is ingested. Returns empty result set with full DDL column shape.

select
    cast(null as string)                           as land_degradation_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as unit_key,
    cast(null as numeric)                          as pct_degraded,
    cast(null as numeric)                          as upper_bound,
    cast(null as numeric)                          as lower_bound,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy) where false
