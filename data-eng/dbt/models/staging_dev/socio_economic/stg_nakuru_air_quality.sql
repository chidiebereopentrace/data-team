{{ config(materialized='table') }}

with base as (
    select
        sensor_id,
        sensor_type,
        location,
        lat as latitude,
        lon as longitude,
        timestamp as observation_timestamp,
        pm10,
        pm2_5,
        humidity_pct,
        temp_c as temperature_c,
        source_archive,
        'nakuru_air_quality_archive' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'nakuru_air_quality_archive') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['sensor_id', 'observation_timestamp']) }} as nakuru_air_quality_sk,
    base.*
from base
