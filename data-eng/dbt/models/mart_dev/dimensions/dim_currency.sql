{{ config(materialized='table') }}

-- Gold: currency dimension from FEWS market price data (distinct currencies observed).

with currencies as (
    select distinct
        common_currency                            as currency_code
    from {{ source('landing', 'FEWS_NET_market_Prices_time_series_data') }}
    where common_currency is not null
)

select
    to_hex(md5(coalesce(currency_code, '')))       as currency_key,
    currency_code,
    cast(null as string)                           as currency_name

from currencies
