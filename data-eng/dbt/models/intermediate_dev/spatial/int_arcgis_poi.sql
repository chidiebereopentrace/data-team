{{ config(materialized='table') }}

select
    objectid,
    osm_id,
    name,
    city,
    country,
    amenity,
    attraction,
    historic,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_arcgis_poi') }}
