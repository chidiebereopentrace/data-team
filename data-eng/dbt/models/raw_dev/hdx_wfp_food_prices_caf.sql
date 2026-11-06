{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'hdx_wfp_food_prices_caf') }}
