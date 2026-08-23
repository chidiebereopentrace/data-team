{{ config(materialized='table') }}

select
    e.time,
    e.valid_time,
    e.latitude,
    e.longitude,
    e.ensemble_member,
    e.temperature_2m,
    g.geo_key,
    e.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_copernicus_era5') }} e
left join {{ ref('int_geography_conformed') }} g
    on g.geo_level = 'city'
   and round(e.latitude, 2) = round(g.latitude, 2)
   and round(e.longitude, 2) = round(g.longitude, 2)
where e.longitude between -25 and 60
  and e.latitude between -35 and 38
  and e.temperature_2m is not null
