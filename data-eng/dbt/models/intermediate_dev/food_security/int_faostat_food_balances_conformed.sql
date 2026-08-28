{{ config(materialized='table') }}

{# Long-form FAOSTAT food balances — one row per area/item/element/year/source. #}

select
    area_code,
    area_code_m49,
    country_name,
    item_code,
    item_code_fbs,
    item_code_cpc,
    product_name,
    element_code,
    element,
    year,
    unit,
    safe_cast(value as float64) as value,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_food_balances') }}
where year is not null
  and value is not null
