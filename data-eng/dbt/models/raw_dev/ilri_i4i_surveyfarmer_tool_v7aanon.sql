{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_i4i_surveyfarmer_tool_v7aanon') }}
