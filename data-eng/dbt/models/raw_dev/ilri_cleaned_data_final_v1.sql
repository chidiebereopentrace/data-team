{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_cleaned_data_final_v1') }}
