{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_explanation_of_calculations_and_outputs') }}
