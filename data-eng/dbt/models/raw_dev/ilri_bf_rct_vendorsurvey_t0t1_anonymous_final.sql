{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_bf_rct_vendorsurvey_t0t1_anonymous_final') }}
