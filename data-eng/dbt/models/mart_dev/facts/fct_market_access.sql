{{ config(materialized='table') }}

-- Gold: market access fact extended from staging_dev fact_market_access.

select
    fact_market_access_key                         as market_access_key,
    geography_key                                  as geo_key,
    period_key                                     as time_key,
    source_key,
    cast(null as string)                           as market_key,
    cast(null as string)                           as audit_key,
    market_access_value                            as access_index,
    cast(null as numeric)                          as distance_to_market,
    cast(null as numeric)                          as travel_time_minutes,
    cast(null as string)                           as purchase_frequency,
    cast(null as int64)                            as outlet_choice_cleanliness,
    cast(null as int64)                            as outlet_choice_hygiene,
    cast(null as int64)                            as outlet_choice_quality,
    cast(null as int64)                            as outlet_choice_price,
    cast(null as int64)                            as years_operating,
    cast(null as int64)                            as operating_days_per_week,
    cast(null as numeric)                          as avg_daily_sales_volume,
    cast(null as string)                           as primary_supply_source,
    fact_market_access_key                         as source_record_id

from {{ ref('fact_market_access') }}
