{{ config(materialized='table') }}

select
    outlet_id,
    survey_date,
    respondent_age,
    respondent_sex,
    consumer_id,
    respondent_type,
    country,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_ilri_vendor_consumer') }}
