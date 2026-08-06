{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'un_peace_security_records') }}
