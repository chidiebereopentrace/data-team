{{ config(materialized='table') }}

-- Gold: value chain fact extended from staging_dev fact_value_chain.

select
    fact_value_chain_key                           as value_chain_key,
    geography_key                                  as geo_key,
    period_key                                     as time_key,
    source_key,
    cast(null as string)                           as product_key,
    unit_key,
    value_chain_value                              as value_share,
    fact_value_chain_key                           as source_record_id

from {{ ref('fact_value_chain') }}
