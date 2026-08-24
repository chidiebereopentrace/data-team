{{ config(materialized='table') }}

select
    country,
    country_code,
    admin_1,
    admin_2,
    market_name,
    product_name,
    cpcv2,
    price_type,
    unit,
    currency,
    value as price_value,
    common_unit_price,
    common_currency_price,
    year,
    month,
    'fews' as price_source,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_fews_market_prices') }}
where value is not null

union all

select
    country,
    cast(null as string) as country_code,
    admin1_name as admin_1,
    cast(null as string) as admin_2,
    market_name,
    product_name,
    cast(null as string) as cpcv2,
    price_type,
    unit,
    currency,
    value as price_value,
    cast(null as float64) as common_unit_price,
    cast(null as float64) as common_currency_price,
    year,
    month,
    'wfp' as price_source,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_wfp_vampire_prices') }}
where value is not null

union all

select
    country_name as country,
    cast(null as string) as country_code,
    cast(null as string) as admin_1,
    cast(null as string) as admin_2,
    cast(null as string) as market_name,
    product_name,
    item_code_cpc as cpcv2,
    element as price_type,
    unit,
    currency,
    safe_cast(value as float64) as price_value,
    cast(null as float64) as common_unit_price,
    cast(null as float64) as common_currency_price,
    year,
    safe_cast(months_code as int64) as month,
    'faostat' as price_source,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_prices') }}
where value is not null
  and safe_cast(value as float64) is not null
