{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'unccd_land_degradation_bulk') }}
