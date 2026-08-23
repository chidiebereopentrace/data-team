{{ config(materialized='table') }}

select
    objectid,
    grid_id,
    ndvi_mean,
    ndvi_mean_secondary,
    shape_area,
    shape_length,
    geometry_wkt,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_vegetation_ndvi') }}
where ndvi_mean is not null
