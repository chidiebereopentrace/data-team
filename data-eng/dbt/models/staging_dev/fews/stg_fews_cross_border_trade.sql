{{ config(materialized='table') }}

with base as (
    select
        reporting_country as country,
        reporting_country_code as country_code,
        border_point,
        source as source_country,
        source_country_code,
        destination as destination_country,
        destination_country_code,
        product as product_name,
        cpcv2,
        flow_type as trade_flow,
        trade_type,
        unit,
        value,
        common_unit_quantity,
        extract(year from cast(period_date as date)) as year,
        extract(month from cast(period_date as date)) as month,
        'FEWS_NET_cross_border_Trade' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'FEWS_NET_cross_border_Trade_time_series_data') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['country', 'border_point', 'product_name', 'trade_flow', 'year', 'month', 'source_natural_key']) }} as fews_cross_border_trade_sk,
    base.*
from base
