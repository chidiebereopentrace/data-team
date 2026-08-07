{{ config(materialized='table') }}

-- Gold: employment fact — schema registered; populated once a labour/employment
-- source is ingested. Returns empty result set with full DDL column shape.

select
    cast(null as string)                           as employment_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as person_key,
    cast(null as string)                           as unit_key,
    cast(null as numeric)                          as employment_count,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy) where false
