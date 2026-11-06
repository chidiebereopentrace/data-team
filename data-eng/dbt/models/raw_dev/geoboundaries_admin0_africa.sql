{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'geoboundaries_admin0_africa') }}
