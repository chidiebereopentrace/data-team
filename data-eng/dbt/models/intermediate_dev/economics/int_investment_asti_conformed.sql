{{ config(materialized='table') }}

select
    area_code,
    area_code_m49,
    country_name,
    donor_code,
    donor,
    purpose_code,
    purpose,
    item_code,
    item,
    element_code,
    element,
    indicator_code,
    indicator,
    institution_code,
    institution,
    degree_code,
    degree,
    sex_code,
    sex,
    cost_category_code,
    cost_category,
    year,
    unit,
    safe_cast(value as float64) as value,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_investment_asti') }}
where year is not null
