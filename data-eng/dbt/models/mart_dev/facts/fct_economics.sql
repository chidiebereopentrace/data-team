{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(cast(m.area_code as string), '') || '|' ||
            coalesce(cast(m.item_code as string), '') || '|' ||
            coalesce(cast(m.element_code as string), '') || '|' ||
            coalesce(m.item, '') || '|' ||
            coalesce(m.element, '') || '|' ||
            coalesce(m.measurement_form, '') || '|' ||
            cast(m.year as string) || '|' ||
            m.source_natural_key
        )) as economics_key,
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'iso.country_iso3') }} as country_iso3,
        i.item_key,
        el.element_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        lower(replace(coalesce(m.item, m.element, m.measurement_form, ''), ' ', '_')) as metric,
        s.source_key as source_id,
        m.measurement_form,
        m.year,
        format_date('%Y%m%d', date(m.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 'm.year', 'cast(null as int64)', 'm.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'm.year', 'cast(null as int64)') }} as as_of_date_basis,
        m.unit,
        m.value,
        m.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_faostat_macro_conformed') }} m
    left join {{ ref('int_faostat_area_iso') }} iso
        on iso.area_code = cast(m.area_code as string)
    left join {{ ref('dim_geography') }} g
        on g.geography_key = m.geo_key
    left join {{ ref('dim_item') }} i
        on lower(i.item_name) = lower(m.item)
    left join {{ ref('dim_element') }} el
        on lower(el.element_name) = lower(m.element)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = m.source_natural_key

    union all

    select
        to_hex(md5(
            coalesce(p.country_iso3, '') || '|gdp_ppp|' ||
            cast(p.year as string) || '|' || p.source_natural_key
        )),
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'p.country_iso3') }} as country_iso3,
        cast(null as string) as item_key,
        cast(null as string) as element_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        'gdp_ppp' as metric,
        s.source_key as source_id,
        'current_price' as measurement_form,
        p.year,
        format_date('%Y%m%d', date(p.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 'p.year', 'cast(null as int64)', 'p.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'p.year', 'cast(null as int64)') }} as as_of_date_basis,
        'PPP USD per capita' as unit,
        p.gdp_per_capita_ppp as value,
        p.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_gdp_ppp_conformed') }} p
    left join {{ ref('dim_geography') }} g
        on g.geography_key = p.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = p.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by economics_key
    order by value desc nulls last
) = 1
