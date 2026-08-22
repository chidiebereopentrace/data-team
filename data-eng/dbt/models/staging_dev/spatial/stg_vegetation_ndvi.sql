{{ config(materialized='table') }}

select
    {{ dbt_utils.generate_surrogate_key(['grid_id']) }} as vegetation_ndvi_sk,
    objectid,
    grid_id,
    mean as ndvi_mean,
    mean_1 as ndvi_mean_secondary,
    shape__area as shape_area,
    shape__length as shape_length,
    geometry_wkt,
    'arcgis_vegetation_ndvi' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'arcgis_vegetation_ndvi') }}
