{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_04_p_q_functions') }}
