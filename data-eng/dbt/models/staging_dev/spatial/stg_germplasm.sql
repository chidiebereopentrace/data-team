{{ config(materialized='table') }}

with crop as (
    select
        c.id as germplasm_id,
        c.taxon,
        c.objectid,
        coalesce(a.geometry_wkt, cast(null as string)) as geometry_wkt,
        coalesce(
            {{ geo_geog_centroid_latitude('c.geography') }},
            {{ geo_centroid_latitude('a.geometry_wkt') }}
        ) as latitude,
        coalesce(
            {{ geo_geog_centroid_longitude('c.geography') }},
            {{ geo_centroid_longitude('a.geometry_wkt') }}
        ) as longitude,
        'crop_germplasm_africa' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'crop_germplasm_africa') }} c
    left join {{ source('raw_dev', 'arcgis_layer_rice_germplasm_in_africa_3d2a9') }} a
        on c.objectid = a.objectid
),

arcgis as (
    select
        id as germplasm_id,
        taxon,
        objectid,
        geometry_wkt,
        {{ geo_centroid_latitude('geometry_wkt') }} as latitude,
        {{ geo_centroid_longitude('geometry_wkt') }} as longitude,
        'arcgis_layer_rice_germplasm_in_africa_3d2a9' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'arcgis_layer_rice_germplasm_in_africa_3d2a9') }}
),

base as (
    select * from crop
    union all
    select * from arcgis
)

select
    {{ dbt_utils.generate_surrogate_key(['germplasm_id', 'source_natural_key']) }} as germplasm_sk,
    base.*
from base
