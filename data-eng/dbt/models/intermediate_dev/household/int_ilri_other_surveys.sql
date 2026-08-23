{{ config(materialized='table') }}

select
    household_id,
    country,
    village,
    survey_type,
    survey_start,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_ilri_other_surveys') }}
