{{ config(materialized='table') }}

select
    objectid,
    protected_area_name,
    feature_count,
    area_protected,
    geometry_wkt,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_protected_areas') }}
