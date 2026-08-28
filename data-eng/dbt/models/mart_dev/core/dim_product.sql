{{ config(materialized='table') }}

select
    to_hex(md5(coalesce(src.item_code, '') || '|' || src.product_name_norm)) as product_key,
    src.item_code,
    any_value(src.product_name) as product_name,
    any_value(src.cpcv2) as cpcv2,
    current_timestamp() as loaded_at
from (
    select
        cast(item_code as string) as item_code,
        product_name,
        lower(trim(product_name)) as product_name_norm,
        cast(item_code_cpc as string) as cpcv2
    from {{ ref('int_faostat_production_conformed') }}
    union all
    select
        cast(null as string) as item_code,
        product as product_name,
        lower(trim(product)) as product_name_norm,
        cast(null as string) as cpcv2
    from {{ ref('int_yield_raw_enriched') }}
    union all
    select
        cpcv2 as item_code,
        product_name,
        lower(trim(product_name)) as product_name_norm,
        cpcv2
    from {{ ref('int_prices_harmonised') }}
    where product_name is not null
    union all
    select
        cast(item_code as string) as item_code,
        product_name,
        lower(trim(product_name)) as product_name_norm,
        cast(item_code_cpc as string) as cpcv2
    from {{ ref('int_faostat_food_balances_conformed') }}
    where product_name is not null
    union all
    select
        cast(item_code as string) as item_code,
        product_name,
        lower(trim(product_name)) as product_name_norm,
        cast(item_code_cpc as string) as cpcv2
    from {{ ref('int_faostat_trade_conformed') }}
    where product_name is not null
    union all
    select
        cast(cpcv2 as string) as item_code,
        product_name,
        lower(trim(product_name)) as product_name_norm,
        cast(cpcv2 as string) as cpcv2
    from {{ ref('int_fews_cross_border_trade_conformed') }}
    where product_name is not null
) src
where src.product_name is not null
group by
    src.item_code,
    src.product_name_norm
