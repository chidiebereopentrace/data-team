{{ config(materialized='table') }}

select
    country,
    alpha_3_code as country_code,
    year,
    index as hdi_value,
    'africa_Human_development_index' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'africa_Human_development_index') }}
