{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'isric_africa_soil_data') }}
