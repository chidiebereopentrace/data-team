{{ config(materialized='table') }}

select
    v.latitude,
    v.longitude,
    v.survey_date,
    v.quantity_trees,
    v.quantity_shrubs,
    v.quantity_grass,
    v.palatability_trees,
    v.palatability_shrubs,
    v.palatability_grass,
    v.carrying_capacity,
    v.currently_grazing,
    v.household_id,
    v.agro_climatic_zone_id,
    v.record_type,
    g.geo_key,
    v.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_ilri_vegetation_feed') }} v
left join {{ ref('int_geography_conformed') }} g
    on g.geo_level = 'city'
   and round(v.latitude, 2) = round(g.latitude, 2)
   and round(v.longitude, 2) = round(g.longitude, 2)
where v.longitude between -25 and 60
  and v.latitude between -35 and 38
