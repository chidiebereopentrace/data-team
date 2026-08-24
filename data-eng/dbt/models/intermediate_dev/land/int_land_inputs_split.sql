{{ config(materialized='table') }}

select
    area_code,
    area_code_m49,
    country_name,
    partner_country_code,
    partner_countries,
    item_code,
    item_code_cpc,
    item,
    months_code,
    months,
    element_code,
    element,
    year,
    unit,
    safe_cast(value as float64) as value,
    case
        when partner_country_code is not null then 'trade'
        when months_code is not null then 'temperature'
        else 'use_or_cover'
    end as input_grain,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_land_inputs') }}
where year is not null
