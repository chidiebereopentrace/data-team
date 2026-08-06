{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_acaricide_use_final_analysis_dataverse') }}
