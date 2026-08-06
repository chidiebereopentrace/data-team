{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_amuse_eth_data') }}
