{{ config(materialized='table') }}

with base as (
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
)

select
    {{ dbt_utils.generate_surrogate_key(['latitude', 'longitude', 'time', 'valid_time', 'ensemble_member', 'step']) }} as copernicus_era5_sk,
    base.*
from base
