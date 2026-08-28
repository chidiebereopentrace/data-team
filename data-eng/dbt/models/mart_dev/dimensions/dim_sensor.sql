{{ config(materialized='table') }}

-- Gold: sensor dimension for air quality and climate monitoring devices.
-- Sourced from Nakuru air quality sensor metadata.

with sensors as (
    select distinct
        cast(sensor_id as string)                  as sensor_id,
        sensor_type,
        cast(location as string)                   as location_description
    from {{ source('raw_dev', 'nakuru_air_quality_archive') }}
    where sensor_id is not null
)

select
    to_hex(md5(coalesce(sensor_id, '')))           as sensor_key,
    sensor_id,
    sensor_type,
    location_description  -- cast(location as string) from source

from sensors
