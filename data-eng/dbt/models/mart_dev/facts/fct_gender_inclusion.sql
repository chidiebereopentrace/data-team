{{ config(materialized='table') }}

-- Gold: gender inclusion fact — schema registered; populated once ILRI household
-- staging is wired in. Returns empty result set with full DDL column shape.

select
    cast(null as string)                           as gender_inclusion_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as person_key,
    cast(null as string)                           as audit_key,
    cast(null as numeric)                          as male_control_share,
    cast(null as numeric)                          as female_control_share,
    cast(null as numeric)                          as male_youth_control_share,
    cast(null as numeric)                          as female_youth_control_share,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy) where false
