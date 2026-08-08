{{ config(materialized='table') }}

-- Gold: prices fact from FEWS market prices time series.

select
    to_hex(md5(concat(
        coalesce(country_code, ''), '|',
        coalesce(market, ''), '|',
        coalesce(product, ''), '|',
        coalesce(period_date, '')
    )))                                            as price_key,

    to_hex(md5(concat(
        coalesce(country_code, ''), '|',
        coalesce(country, ''), '|',
        coalesce(admin_1, '')
    )))                                            as geo_key,

    cast(left(coalesce(period_date, '2000'), 4) as int64)
                                                   as time_key,

    to_hex(md5('FEWS_NET_market_Prices_time_series_data'))
                                                   as source_key,

    to_hex(md5(coalesce(product, '')))             as product_key,

    to_hex(md5(concat(
        coalesce(market, ''), '|', coalesce(country_code, '')
    )))                                            as market_key,

    to_hex(md5(coalesce(unit, '')))                as unit_key,
    cast(null as string)                           as audit_key,

    cast(value as numeric)                         as value,
    cast(common_currency_price as numeric)         as common_currency_price,
    cast(exchange_rate as numeric)                 as exchange_rate,
    cast(pct_change_from_one_month_ago as numeric) as pct_change_1m,
    cast(pct_change_from_one_year_ago as numeric)  as pct_change_1y,
    to_hex(md5(concat(
        coalesce(country_code, ''), '|', coalesce(market, ''), '|',
        coalesce(product, ''), '|', coalesce(period_date, '')
    )))                                            as source_record_id

from {{ source('landing', 'FEWS_NET_market_Prices_time_series_data') }}
where value is not null

{% if is_incremental() %}
  and period_date > (
    select cast(max(time_key) as string) from {{ this }}
  )
{% endif %}
