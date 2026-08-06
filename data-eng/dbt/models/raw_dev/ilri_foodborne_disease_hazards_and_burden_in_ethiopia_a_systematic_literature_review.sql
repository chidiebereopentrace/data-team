{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_foodborne_disease_hazards_and_burden_in_ethiopia_a_systematic_literature_review') }}
