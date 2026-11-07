{{ config(materialized='table') }}

-- Gold: land use fact extended from staging_dev fact_land_use.

select
    fact_land_use_key                              as land_use_fact_key,
    geography_key                                  as geo_key,
    period_key                                     as time_key,
    source_key,
    crop_key                                       as land_use_key,
    land_use_value                                 as area,
    fact_land_use_key                              as source_record_id

from {{ ref('fact_land_use') }}
