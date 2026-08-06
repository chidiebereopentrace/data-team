{{ config(materialized='table') }}

select
    time,
    valid_time,
    latitude,
    longitude,
    number as ensemble_member,
    step,
    surface,
    t2m as temperature_2m,
    'Copernicus_ERA5' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'copernicus_climate_raw_era5_stats_2023_06') }}
