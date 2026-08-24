{{ config(materialized='table') }}

select
    wardid,
    provincena as province_name,
    localmunic as municipality,
    districtmu as district,
    year,
    latitude,
    longitude,
    employed,
    unemployed,
    unemploypc,
    total as population_total,
    geometry_wkt,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_arcgis_sa_wards_demographics') }}
