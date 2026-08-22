{{ config(materialized='table') }}

select
    {{ dbt_utils.generate_surrogate_key(['objectid']) }} as protected_areas_sk,
    objectid,
    name as protected_area_name,
    count as feature_count,
    analysisarea as area_protected,
    geometry_wkt,
    'arcgis_land_protected_areas' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'arcgis_land_protected_areas') }}
