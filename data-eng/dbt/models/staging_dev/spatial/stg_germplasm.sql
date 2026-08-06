{{ config(materialized='table') }}

select
    id as germplasm_id,
    taxon,
    objectid,
    geography,
    cast(null as string) as geometry_wkt,
    'crop_germplasm_africa' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'crop_germplasm_africa') }}

union all

select
    id as germplasm_id,
    taxon,
    objectid,
    cast(null as geography) as geography,
    geometry_wkt,
    'arcgis_layer_rice_germplasm_in_africa_3d2a9' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'arcgis_layer_rice_germplasm_in_africa_3d2a9') }}
