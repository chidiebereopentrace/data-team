{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_05_nutrient_composition') }}
