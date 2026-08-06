{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_dgea1_calvinfo_public') }}
