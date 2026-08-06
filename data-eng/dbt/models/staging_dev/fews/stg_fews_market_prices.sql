{{ config(materialized='table') }}

select
    country,
    country_code,
    admin_1,
    admin_2,
    market as market_name,
    product as product_name,
    cpcv2,
    price_type,
    unit,
    currency,
    value,
    common_unit_price,
    common_currency_price,
    extract(year from cast(period_date as date)) as year,
    extract(month from cast(period_date as date)) as month,
    'FEWS_NET_market_Prices' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'FEWS_NET_market_Prices_time_series_data') }}
