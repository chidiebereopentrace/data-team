{{ config(materialized='table') }}

select
    country,
    country_code,
    border_point,
    source_country,
    source_country_code,
    destination_country,
    destination_country_code,
    product_name,
    cpcv2,
    trade_flow,
    trade_type,
    unit,
    safe_cast(value as float64) as value,
    common_unit_quantity,
    year,
    month,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_fews_cross_border_trade') }}
where year is not null
