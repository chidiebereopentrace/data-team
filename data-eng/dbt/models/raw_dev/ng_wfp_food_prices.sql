{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ng_wfp_food_prices') }}
