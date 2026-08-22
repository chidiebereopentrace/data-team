{{ config(materialized='table') }}

with base as (
    select
        *,
        'arcgis_south_africa_wards_demographics_2ce07' as source_natural_key
    from {{ source('raw_dev', 'arcgis_south_africa_wards_demographics_2ce07') }}
)

select
    to_hex(md5(to_json_string(base))) as arcgis_sa_wards_demographics_sk,
    base.*,
    current_timestamp() as loaded_at
from base
