{{ config(materialized='table') }}

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
