{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

with {{ dim_country_by_native_id_cte() }},
{{ dim_country_by_iso3_cte() }},

base as (
    select
        to_hex(md5(
            coalesce(cast(m.area_code as string), '') || '|' ||
            coalesce(m.item, '') || '|' ||
            coalesce(m.element, '') || '|' ||
            coalesce(cast(m.year as string), '') || '|' ||
            coalesce(m.source_natural_key, '')
        )) as machinery_key,
        coalesce(g_nat.geography_key, g_iso.geography_key) as geography_key,
        coalesce(g_nat.geo_level, g_iso.geo_level) as geo_level,
        {{ acf_country_iso3('coalesce(g_nat.country_iso3, g_iso.country_iso3)', 'iso.country_iso3') }} as country_iso3,
        i.item_key,
        el.element_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as data_level,
        {{ acf_geo_scope_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as geo_scope,
        {{ acf_place_scope_coalesce('g_nat', 'g_iso', 'iso.country_iso3', 'm.country_name') }} as place_scope,
        lower(replace(coalesce(m.element, m.item, ''), ' ', '_')) as metric,
        s.source_key as source_id,
        m.country_name,
        m.item,
        m.element,
        m.year,
        format_date('%Y%m%d', date(m.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 'm.year', 'cast(null as int64)', 'm.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'm.year', 'cast(null as int64)') }} as as_of_date_basis,
        m.unit,
        m.value,
        m.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_discontinued_machinery') }} m
    left join {{ ref('int_faostat_area_iso') }} iso
        on iso.area_code = cast(m.area_code as string)
    left join country_by_native_id g_nat
        on g_nat.native_id = cast(m.area_code as string)
    left join country_by_iso3 g_iso
        on upper(trim(g_iso.country_iso3)) = upper(trim(iso.country_iso3))
    left join {{ ref('dim_item') }} i
        on lower(i.item_name) = lower(m.item)
    left join {{ ref('dim_element') }} el
        on lower(el.element_name) = lower(m.element)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = m.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by machinery_key
    order by value desc nulls last
) = 1
