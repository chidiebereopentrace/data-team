{{ config(materialized='table') }}

select
    area_code,
    area_code_m49,
    country_name,
    item_code,
    item_code_cpc,
    product_name,
    year,
    source_natural_key,
    max(case when lower(element) = 'area harvested' then safe_cast(value as float64) end) as area_harvested,
    max(case when lower(element) = 'production' then safe_cast(value as float64) end) as production_qty,
    max(case when lower(element) = 'yield' then safe_cast(value as float64) end) as yield_value,
    max(case when lower(element) = 'area harvested' then unit end) as area_unit,
    max(case when lower(element) = 'production' then unit end) as production_unit,
    max(case when lower(element) = 'yield' then unit end) as yield_unit,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_production') }}
group by
    area_code,
    area_code_m49,
    country_name,
    item_code,
    item_code_cpc,
    product_name,
    year,
    source_natural_key
