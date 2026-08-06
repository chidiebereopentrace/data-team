{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_02_feed_price_modeled_impact') }}
