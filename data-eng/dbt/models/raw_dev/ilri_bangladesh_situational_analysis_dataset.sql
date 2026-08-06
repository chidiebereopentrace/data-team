{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_bangladesh_situational_analysis_dataset') }}
