{{ config(materialized='table') }}

with base as (
    select
        id,
        country,
        fewsnet_region,
        geographic_group,
        fnid,
        cast(null as string) as market,
        cast(null as string) as market_name,
        cast(null as string) as product,
        cast(null as string) as product_name,
        cast(null as string) as cpcv2,
        cast(null as string) as unit,
        cast(null as string) as currency,
        cast(null as string) as price_type,
        cast(null as string) as phase,
        cast(null as string) as phase_name,
        cast(null as string) as scenario_name,
        cast(null as string) as indicator_name,
        data_source_organization,
        first_period_date,
        last_period_date,
        datapoint_count,
        'food_security_classifications' as fews_domain,
        'FEWS_NET_food_security_classifications_data_series' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'FEWS_NET_food_security_classifications_data_series') }}

    union all

    select
        id,
        country,
        fewsnet_region,
        geographic_group,
        fnid,
        cast(null as string) as market,
        cast(null as string) as market_name,
        cast(null as string) as product,
        cast(null as string) as product_name,
        cast(null as string) as cpcv2,
        cast(null as string) as unit,
        cast(null as string) as currency,
        cast(null as string) as price_type,
        phase,
        phase_name,
        scenario_name,
        cast(null as string) as indicator_name,
        data_source_organization,
        first_period_date,
        last_period_date,
        datapoint_count,
        'population_estimates' as fews_domain,
        'FEWS_NET_Food_insecure_population_estimates_data_series' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'FEWS_NET_Food_insecure_population_estimates_data_series') }}

    union all

    select
        id,
        country,
        fewsnet_region,
        geographic_group,
        fnid,
        cast(market as string) as market,
        market_name,
        cast(product as string) as product,
        product_name,
        cpcv2,
        unit,
        currency,
        price_type,
        cast(null as string) as phase,
        cast(null as string) as phase_name,
        cast(null as string) as scenario_name,
        cast(null as string) as indicator_name,
        data_source_organization,
        first_period_date,
        last_period_date,
        datapoint_count,
        'market_prices' as fews_domain,
        'FEWS_NET_market_prices_data_series' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'FEWS_NET_market_prices_data_series') }}

    union all

    select
        id,
        reporting_country as country,
        cast(null as string) as fewsnet_region,
        cast(null as string) as geographic_group,
        cast(null as string) as fnid,
        cast(null as string) as market,
        border_point_name as market_name,
        cast(product as string) as product,
        product_name,
        cpcv2,
        unit,
        cast(null as string) as currency,
        cast(null as string) as price_type,
        cast(null as string) as phase,
        cast(null as string) as phase_name,
        cast(null as string) as scenario_name,
        cast(null as string) as indicator_name,
        data_source_organization,
        first_period_date,
        last_period_date,
        datapoint_count,
        'cross_border_trade' as fews_domain,
        'FEWS_NET_cross_border_trade_data_series' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'FEWS_NET_cross_border_trade_data_series') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['id', 'source_natural_key']) }} as fews_data_series_catalogue_sk,
    base.*
from base
