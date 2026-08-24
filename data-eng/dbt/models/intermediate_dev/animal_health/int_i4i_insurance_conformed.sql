{{ config(materialized='table') }}

select
    household_id,
    country,
    region_id,
    farmer_location,
    farmer_category,
    herd_size_category,
    insurance_start_year,
    local_currency,
    record_type,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_ilri_i4i_livestock_insurance') }}
where household_id is not null
