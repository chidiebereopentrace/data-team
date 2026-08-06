{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_03_feed_price_quality_data') }}
