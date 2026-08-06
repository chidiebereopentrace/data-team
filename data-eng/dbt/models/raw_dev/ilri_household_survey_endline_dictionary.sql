{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_household_survey_endline_dictionary') }}
