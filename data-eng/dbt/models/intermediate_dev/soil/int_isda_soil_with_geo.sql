{{ config(materialized='table') }}

select
    s.longitude,
    s.latitude,
    coalesce(s.country, g.country_name) as country_name,
    g.country_iso2,
    coalesce(s.city, g.city_name) as city_name,
    g.geo_key,
    s.soil_property,
    s.depth,
    s.value,
    s.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_isda_soil_enriched') }} s
left join {{ ref('int_geography_conformed') }} g
    on g.geo_level = 'city'
   and round(s.latitude, 2) = round(g.latitude, 2)
   and round(s.longitude, 2) = round(g.longitude, 2)
where s.longitude between -25 and 60
  and s.latitude between -35 and 38
  and not (s.longitude = 0 and s.latitude = 0)
