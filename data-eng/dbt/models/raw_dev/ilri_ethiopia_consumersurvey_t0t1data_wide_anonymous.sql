{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_ethiopia_consumersurvey_t0t1data_wide_anonymous') }}
