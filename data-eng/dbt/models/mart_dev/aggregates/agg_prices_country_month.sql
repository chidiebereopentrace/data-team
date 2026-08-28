{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(g.geography_key, '') || '|' ||
        coalesce(p.product_key, '') || '|' ||
        coalesce(p.price_type, '') || '|' ||
        coalesce(p.price_source, '') || '|' ||
        coalesce(p.source_key, '') || '|' ||
        coalesce(cast(p.year as string), '') || '|' ||
        coalesce(cast(p.month as string), '')
    )) as agg_prices_country_month_key,
    g.geography_key,
    case
        when p.month between 1 and 12
            then format_date('%Y%m%d', date(p.year, p.month, 1))
        when p.year is not null
            then format_date('%Y%m%d', date(p.year, 1, 1))
    end as date_key,
    g.country_name,
    p.product_key,
    p.price_type,
    p.price_source,
    p.source_key,
    p.year,
    p.month,
    avg(p.value) as price_avg,
    avg(p.common_unit_price) as common_unit_price_avg,
    count(*) as market_count,
    current_timestamp() as loaded_at
from {{ ref('fct_prices') }} p
left join {{ ref('dim_geography') }} g
    on g.geography_key = p.geography_key
group by
    g.geography_key,
    g.country_name,
    p.product_key,
    p.price_type,
    p.price_source,
    p.source_key,
    p.year,
    p.month
