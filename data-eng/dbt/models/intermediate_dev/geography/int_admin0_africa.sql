{{ config(materialized='table') }}

{# geoBoundaries ADM0 polygons (point ST_CONTAINS) + city-bbox envelopes (grid coverage fallback). #}

with geoboundaries as (
    select
        upper(trim(country_iso2)) as country_iso2,
        upper(trim(country_iso3)) as country_iso3,
        trim(country_name) as country_name,
        geog,
        min_lat,
        max_lat,
        min_lng,
        max_lng,
        1 as source_priority
    from {{ source('raw_dev', 'geoboundaries_admin0_africa') }}
    where geog is not null
      and country_iso2 is not null
),

city_bounds as (
    select
        upper(trim(iso2)) as country_iso2,
        upper(trim(iso3)) as country_iso3,
        trim(country) as country_name,
        min(safe_cast(lat as float64)) as min_lat,
        max(safe_cast(lat as float64)) as max_lat,
        min(safe_cast(lng as float64)) as min_lng,
        max(safe_cast(lng as float64)) as max_lng
    from {{ source('staging_dev', 'stg_geo') }}
    where iso2 is not null
      and lat is not null
      and lng is not null
    group by 1, 2, 3
),

city_padded as (
    select
        country_iso2,
        country_iso3,
        country_name,
        min_lat - greatest(max_lat - min_lat, 0.1) / 2 as pad_min_lat,
        max_lat + greatest(max_lat - min_lat, 0.1) / 2 as pad_max_lat,
        min_lng - greatest(max_lng - min_lng, 0.1) / 2 as pad_min_lng,
        max_lng + greatest(max_lng - min_lng, 0.1) / 2 as pad_max_lng
    from city_bounds
),

city_envelopes as (
    select
        country_iso2,
        country_iso3,
        country_name,
        st_geogfromtext(
            format(
                'POLYGON((%f %f, %f %f, %f %f, %f %f, %f %f))',
                pad_min_lng, pad_min_lat,
                pad_max_lng, pad_min_lat,
                pad_max_lng, pad_max_lat,
                pad_min_lng, pad_max_lat,
                pad_min_lng, pad_min_lat
            )
        ) as geog,
        pad_min_lat as min_lat,
        pad_max_lat as max_lat,
        pad_min_lng as min_lng,
        pad_max_lng as max_lng,
        2 as source_priority
    from city_padded
    where pad_max_lat > pad_min_lat
      and pad_max_lng > pad_min_lng
),

scoped_geoboundaries as (
    select g.*
    from geoboundaries g
    inner join {{ ref('ref_m49_country') }} rc
        on rc.country_iso2 = g.country_iso2
    where rc.in_africa_scope
       or rc.country_iso2 in ('YT', 'RE', 'SH', 'TF', 'EH')
),

scoped_city as (
    select c.*
    from city_envelopes c
    inner join {{ ref('ref_m49_country') }} rc
        on rc.country_iso2 = c.country_iso2
    where rc.in_africa_scope
       or rc.country_iso2 in ('YT', 'RE', 'SH', 'TF', 'EH')
),

unioned as (
    select * from scoped_geoboundaries
    union all
    select * from scoped_city
)

select
    country_iso2,
    country_iso3,
    country_name,
    min_lat,
    max_lat,
    min_lng,
    max_lng,
    geog,
    source_priority,
    current_timestamp() as loaded_at
from unioned
where min_lat is not null
  and max_lat is not null
  and min_lng is not null
  and max_lng is not null
  and max_lat > min_lat
  and max_lng > min_lng
