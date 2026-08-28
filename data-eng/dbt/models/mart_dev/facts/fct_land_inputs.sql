{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['input_grain', 'data_level', 'country_iso3', 'source_key']
) }}

with {{ dim_country_by_native_id_cte() }},
{{ dim_country_by_iso3_cte() }},

base as (
    select
        to_hex(md5(
            coalesce(cast(l.area_code as string), '') || '|' ||
            coalesce(l.partner_country_code, '') || '|' ||
            coalesce(l.item, '') || '|' ||
            coalesce(l.element, '') || '|' ||
            coalesce(l.months_code, '') || '|' ||
            coalesce(l.input_grain, '') || '|' ||
            cast(l.year as string) || '|' ||
            l.source_natural_key
        )) as land_input_key,
        coalesce(g_nat.geography_key, g_iso.geography_key) as geography_key,
        coalesce(g_nat.geo_level, g_iso.geo_level) as geo_level,
        {{ acf_country_iso3('coalesce(g_nat.country_iso3, g_iso.country_iso3)', 'iso.country_iso3') }} as country_iso3,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as data_level,
        {{ acf_geo_scope_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as geo_scope,
        {{ acf_place_scope_coalesce('g_nat', 'g_iso', 'iso.country_iso3', 'l.country_name') }} as place_scope,
        concat(lower(l.input_grain), '_', lower(replace(coalesce(l.element, l.item, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        i.item_key,
        el.element_key,
        l.input_grain,
        l.country_name,
        l.partner_countries,
        l.year,
        format_date('%Y%m%d', date(l.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 'l.year', 'cast(null as int64)', 'l.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'l.year', 'cast(null as int64)') }} as as_of_date_basis,
        l.unit,
        l.value,
        l.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_land_inputs_split') }} l
    left join {{ ref('int_faostat_area_iso') }} iso
        on iso.area_code = cast(l.area_code as string)
    left join country_by_native_id g_nat
        on g_nat.native_id = cast(l.area_code as string)
    left join country_by_iso3 g_iso
        on upper(trim(g_iso.country_iso3)) = upper(trim(iso.country_iso3))
    left join {{ ref('dim_item') }} i
        on lower(i.item_name) = lower(l.item)
    left join {{ ref('dim_element') }} el
        on lower(el.element_name) = lower(l.element)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = l.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by land_input_key
    order by value desc nulls last
) = 1
