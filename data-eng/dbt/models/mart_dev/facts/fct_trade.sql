{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['trade_grain', 'data_level', 'country_iso3', 'source_key']
) }}

with country_by_native_id as (
    select *
    from {{ ref('dim_geography') }}
    where geo_level = 'country'
      and native_id is not null
    qualify row_number() over (
        partition by native_id
        order by population desc nulls last, geography_key
    ) = 1
),

country_by_iso3 as (
    select *
    from {{ ref('dim_geography') }}
    where geo_level = 'country'
      and country_iso3 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso3))
        order by population desc nulls last, geography_key
    ) = 1
),

country_by_name as (
    select *
    from {{ ref('dim_geography') }}
    where geo_level = 'country'
      and country_name is not null
    qualify row_number() over (
        partition by lower(trim(country_name))
        order by population desc nulls last, geography_key
    ) = 1
),

product_by_name as (
    select *
    from {{ ref('dim_product') }}
    where product_name is not null
    qualify row_number() over (
        partition by lower(trim(product_name))
        order by item_code nulls last, product_key
    ) = 1
),

base as (
    select
        to_hex(md5(
            'faostat|' ||
            coalesce(cast(t.area_code as string), '') || '|' ||
            coalesce(cast(t.item_code as string), '') || '|' ||
            coalesce(cast(t.element_code as string), '') || '|' ||
            coalesce(cast(t.year as string), '') || '|' ||
            coalesce(t.source_natural_key, '')
        )) as trade_key,
        coalesce(g_nat.geography_key, g_iso.geography_key) as geography_key,
        coalesce(g_nat.geo_level, g_iso.geo_level) as geo_level,
        {{ acf_country_iso3('coalesce(g_nat.country_iso3, g_iso.country_iso3)', 'iso.country_iso3') }} as country_iso3,
        pr.product_key,
        i.item_key,
        el.element_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as data_level,
        {{ acf_geo_scope_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as geo_scope,
        {{ acf_place_scope_coalesce('g_nat', 'g_iso', 'iso.country_iso3', 't.country_name') }} as place_scope,
        concat('trade_', lower(replace(coalesce(t.element, t.product_name, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        'faostat_country_year' as trade_grain,
        t.country_name,
        cast(null as string) as source_country,
        cast(null as string) as destination_country,
        cast(null as string) as border_point,
        cast(null as string) as trade_flow,
        t.product_name,
        t.element,
        t.year,
        cast(null as int64) as month,
        format_date('%Y%m%d', date(t.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 't.year', 'cast(null as int64)', 't.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 't.year', 'cast(null as int64)') }} as as_of_date_basis,
        t.unit,
        t.value,
        t.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_faostat_trade_conformed') }} t
    left join {{ ref('int_faostat_area_iso') }} iso
        on cast(iso.area_code as string) = cast(t.area_code as string)
    left join country_by_native_id g_nat
        on g_nat.native_id = cast(t.area_code as string)
    left join country_by_iso3 g_iso
        on upper(trim(g_iso.country_iso3)) = upper(trim(iso.country_iso3))
    left join {{ ref('dim_product') }} pr
        on pr.item_code = cast(t.item_code as string)
       and lower(pr.product_name) = lower(t.product_name)
    left join {{ ref('dim_item') }} i
        on lower(i.item_name) = lower(t.product_name)
    left join {{ ref('dim_element') }} el
        on lower(el.element_name) = lower(t.element)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = t.source_natural_key

    union all

    select
        to_hex(md5(
            'fews|' ||
            coalesce(f.country, '') || '|' ||
            coalesce(f.border_point, '') || '|' ||
            coalesce(f.source_country, '') || '|' ||
            coalesce(f.destination_country, '') || '|' ||
            coalesce(f.product_name, '') || '|' ||
            coalesce(f.trade_flow, '') || '|' ||
            coalesce(cast(f.year as string), '') || '|' ||
            coalesce(cast(f.month as string), '') || '|' ||
            coalesce(f.source_natural_key, '')
        )),
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3') }} as country_iso3,
        pr.product_key,
        cast(null as string) as item_key,
        cast(null as string) as element_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        concat('trade_', lower(replace(coalesce(f.product_name, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        'fews_border_month',
        f.country,
        f.source_country,
        f.destination_country,
        f.border_point,
        f.trade_flow,
        f.product_name,
        cast(null as string) as element,
        f.year,
        f.month,
        case
            when f.month between 1 and 12
                then format_date('%Y%m%d', date(f.year, f.month, 1))
            else format_date('%Y%m%d', date(f.year, 1, 1))
        end as date_key,
        {{ acf_as_of_date('cast(null as date)', 'f.year', 'f.month', 'f.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'f.year', 'f.month') }} as as_of_date_basis,
        f.unit,
        f.value,
        f.source_natural_key,
        current_timestamp()
    from {{ ref('int_fews_cross_border_trade_conformed') }} f
    left join country_by_name g
        on lower(trim(g.country_name)) = lower(trim(f.country))
    left join product_by_name pr
        on lower(pr.product_name) = lower(f.product_name)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = f.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by trade_key
    order by value desc nulls last
) = 1
