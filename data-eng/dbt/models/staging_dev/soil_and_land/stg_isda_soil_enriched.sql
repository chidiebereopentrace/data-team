{{ config(materialized='table') }}

with deduped as (
    select distinct
        lon_wgs84 as longitude,
        lat_wgs84 as latitude,
        property as soil_property,
        depth,
        value
    from {{ source('raw_dev', 'enriched_isda_soil_property_Africa') }}
),

geo_attrs as (
    select
        lon_wgs84 as longitude,
        lat_wgs84 as latitude,
        max(country) as country,
        max(city) as city
    from {{ source('raw_dev', 'enriched_isda_soil_property_Africa') }}
    group by 1, 2
),

base as (
    select
        d.longitude,
        d.latitude,
        g.country,
        g.city,
        d.soil_property,
        d.depth,
        d.value,
        'enriched_isda_soil_property_Africa' as source_natural_key,
        current_timestamp() as loaded_at
    from deduped d
    left join geo_attrs g
        on d.longitude = g.longitude
       and d.latitude = g.latitude
)

select
    {{ dbt_utils.generate_surrogate_key(['longitude', 'latitude', 'soil_property', 'depth']) }} as isda_soil_enriched_sk,
    base.*
from base
