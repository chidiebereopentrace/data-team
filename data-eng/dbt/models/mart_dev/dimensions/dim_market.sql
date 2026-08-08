{{ config(materialized='table') }}

-- Gold: market dimension sourced from FEWS market prices data series.
-- Captures market names, locations, and infrastructure attributes.

with fews_markets as (
    select distinct
        market,
        country,
        country_code,
        admin_1,
        admin_2
    from {{ source('landing', 'FEWS_NET_market_Prices_time_series_data') }}
    where market is not null
)

select
    to_hex(md5(concat(
        coalesce(market, ''), '|', coalesce(country_code, '')
    )))                                            as market_key,
    to_hex(md5(concat(
        coalesce(market, ''), '|', coalesce(country_code, '')
    )))                                            as market_natural_key,
    market                                         as market_name,
    'commodity'                                    as market_type,
    to_hex(md5(concat(
        coalesce(country_code, ''), '|',
        coalesce(country, ''), '|',
        coalesce(admin_1, '')
    )))                                            as geo_key,
    cast(null as bool)                             as has_electricity,
    cast(null as bool)                             as has_running_water,
    cast(null as bool)                             as has_handwashing,
    cast(null as bool)                             as has_waste_disposal,
    cast(null as bool)                             as has_refrigeration,
    cast(null as string)                           as outlet_structure_type,
    cast(null as string)                           as floor_condition,
    cast(null as string)                           as infrastructure_adequacy,
    cast(null as string)                           as description,
    true                                           as is_current

from fews_markets
