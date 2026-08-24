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
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_yield_raw_data') }}
where country is not null
