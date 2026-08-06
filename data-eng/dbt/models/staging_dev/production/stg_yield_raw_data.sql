{{ config(materialized='table') }}

select
    fnid,
    country,
    country_code,
    admin_1,
    admin_2,
    product,
    season_name,
    planting_year,
    planting_month,
    harvest_year,
    harvest_month,
    crop_production_system,
    qc_flag,
    area,
    production,
    yield,
    'yield_raw_data' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'yield_raw_data') }}
