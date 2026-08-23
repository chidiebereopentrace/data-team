{{ config(materialized='table') }}

select
    germplasm_id,
    taxon,
    objectid,
    geography,
    geometry_wkt,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_germplasm') }}
