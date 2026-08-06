{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 's4a_field_land_cover_use_info') }}
