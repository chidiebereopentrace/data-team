{{ config(materialized='table') }}

select
    area_code,
    area_code_m49,
    country_name,
    partner_country_code,
    partner_country_code_m49,
    partner_countries,
    item_code,
    item,
    element_code,
    element,
    year,
    unit,
    safe_cast(value as float64) as value,
    case
        when partner_country_code is not null then 'trade_flow'
        else 'production_or_capacity'
    end as forestry_grain,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_forestry') }}
where year is not null
