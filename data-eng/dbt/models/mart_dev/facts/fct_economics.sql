{{ config(materialized='table') }}

-- Gold: economics fact — schema registered; populated once ILRI household
-- staging is wired in. Returns empty result set with full DDL column shape.

select
    cast(null as string)                           as economics_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as unit_key,
    cast(null as string)                           as audit_key,
    cast(null as numeric)                          as total_income,
    cast(null as numeric)                          as farm_income,
    cast(null as numeric)                          as offfarm_income,
    cast(null as numeric)                          as monthly_milk_revenue,
    cast(null as numeric)                          as monthly_feed_cost,
    cast(null as numeric)                          as monthly_vet_cost,
    cast(null as numeric)                          as net_monthly_cashflow,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy) where false
