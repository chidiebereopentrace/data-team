{{ config(materialized='table') }}

select
    sensor_id,
    sensor_type,
    location,
    latitude,
    longitude,
    observation_timestamp,
    pm10,
    pm2_5,
    humidity_pct,
    temperature_c,
    source_archive,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_nakuru_air_quality') }}
where latitude between -35 and 38
  and longitude between -25 and 60
