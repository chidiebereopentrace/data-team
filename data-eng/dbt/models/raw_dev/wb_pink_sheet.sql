{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'wb_pink_sheet') }}
