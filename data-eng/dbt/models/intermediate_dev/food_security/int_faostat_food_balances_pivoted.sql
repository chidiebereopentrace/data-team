{{ config(materialized='table') }}

select
    area_code,
    area_code_m49,
    country_name,
    item_code,
    item_code_fbs,
    item_code_cpc,
    product_name,
    year,
    source_natural_key,
    max(case when lower(element) = 'production' then safe_cast(value as float64) end) as production,
    max(case when lower(element) = 'import quantity' then safe_cast(value as float64) end) as imports,
    max(case when lower(element) = 'export quantity' then safe_cast(value as float64) end) as exports,
    max(case when lower(element) = 'food' then safe_cast(value as float64) end) as food,
    max(case when lower(element) = 'feed' then safe_cast(value as float64) end) as feed,
    max(case when lower(element) = 'losses' then safe_cast(value as float64) end) as losses,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_food_balances') }}
group by
    area_code,
    area_code_m49,
    country_name,
    item_code,
    item_code_fbs,
    item_code_cpc,
    product_name,
    year,
    source_natural_key
