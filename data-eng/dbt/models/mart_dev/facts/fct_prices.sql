{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key', 'product_key']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(p.price_source, '') || '|' ||
            coalesce(p.country, '') || '|' ||
            coalesce(p.admin_1, '') || '|' ||
            coalesce(p.market_name, '') || '|' ||
            coalesce(p.product_name, '') || '|' ||
            coalesce(p.cpcv2, '') || '|' ||
            coalesce(p.price_type, '') || '|' ||
            coalesce(p.unit, '') || '|' ||
            coalesce(p.currency, '') || '|' ||
            coalesce(cast(p.year as string), '') || '|' ||
            coalesce(cast(p.month as string), '') || '|' ||
            coalesce(p.faostat_months_code, '') || '|' ||
            coalesce(p.source_natural_key, '')
        )) as price_key,
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'p.country_iso3') }} as country_iso3,
        m.market_key,
        pr.product_key,
        u.unit_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope_coalesce('g', 'g', 'p.country_iso3', 'p.country') }} as place_scope,
        concat('price_', lower(replace(coalesce(p.price_type, ''), ' ', '_')), '_', lower(replace(coalesce(p.product_name, p.cpcv2, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        p.price_type,
        p.price_source,
        p.year,
        p.month,
        coalesce(p.faostat_months_code, cast(p.month as string)) as month_code,
        case
            when p.month between 1 and 12
                then format_date('%Y%m%d', date(p.year, p.month, 1))
            when p.year is not null
                then format_date('%Y%m%d', date(p.year, 1, 1))
        end as date_key,
        {{ acf_as_of_date('cast(null as date)', 'p.year', 'p.month', 'p.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'p.year', 'p.month') }} as as_of_date_basis,
        p.price_value as value,
        p.unit,
        p.common_unit_price,
        p.common_currency_price,
        p.currency,
        p.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_prices_with_geo') }} p
    left join {{ ref('dim_geography') }} g
        on g.geography_key = p.geo_key
    left join {{ ref('dim_market') }} m
        on lower(m.country) = lower(p.country)
       and lower(m.market_name) = lower(p.market_name)
    left join {{ ref('dim_product') }} pr
        on (p.cpcv2 is not null and pr.item_code = p.cpcv2)
        or (p.cpcv2 is null and lower(pr.product_name) = lower(p.product_name))
    left join {{ ref('dim_unit') }} u
        on lower(u.unit_code) = lower(p.unit)
       and u.unit_type = 'quantity'
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = p.source_natural_key
    where p.year is not null
)

select *
from base
qualify row_number() over (
    partition by price_key
    order by value desc nulls last
) = 1
