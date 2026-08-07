{{ config(materialized='incremental', unique_key='production_key') }}

-- Gold: production fact from staging_dev fact_production (FAOSTAT + yield sources).

select
    fact_production_key                            as production_key,
    geography_key                                  as geo_key,
    period_key                                     as time_key,
    season_key,
    source_key,
    crop_key                                       as product_key,
    cast(null as string)                           as livestock_key,
    cast(null as string)                           as production_system_key,
    unit_key,
    cast(null as string)                           as audit_key,
    cast(null as numeric)                          as area,
    production_value                               as production_qty,
    cast(null as numeric)                          as yield,
    cast(null as numeric)                          as value,
    cast(null as numeric)                          as morning_yield,
    cast(null as numeric)                          as afternoon_yield,
    cast(null as int64)                            as lactation_number,
    fact_production_key                            as source_record_id

from {{ ref('fact_production') }}

{% if is_incremental() %}
  where period_key > (select max(time_key) from {{ this }})
{% endif %}
