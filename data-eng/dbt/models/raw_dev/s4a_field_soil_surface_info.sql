{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 's4a_field_soil_surface_info') }}
