{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_impact_assessment_data_lira_uganda_nov25') }}
