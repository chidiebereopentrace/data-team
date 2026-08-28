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
            coalesce(cast(a.area_code as string), '') || '|' ||
            coalesce(cast(a.item_code as string), '') || '|' ||
            coalesce(cast(a.element_code as string), '') || '|' ||
            coalesce(cast(a.year as string), '') || '|' ||
            coalesce(a.source_natural_key, '')
        )) as humanitarian_key,
        coalesce(g_nat.geography_key, g_iso.geography_key) as geography_key,
        coalesce(g_nat.geo_level, g_iso.geo_level) as geo_level,
        {{ acf_country_iso3('coalesce(g_nat.country_iso3, g_iso.country_iso3)', 'iso.country_iso3') }} as country_iso3,
        i.item_key,
        el.element_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as data_level,
        {{ acf_geo_scope_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as geo_scope,
        {{ acf_place_scope_coalesce('g_nat', 'g_iso', 'iso.country_iso3', 'a.country_name') }} as place_scope,
        lower(replace(coalesce(a.element, a.item, ''), ' ', '_')) as metric,
        s.source_key as source_id,
        a.year,
        format_date('%Y%m%d', date(a.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 'a.year', 'cast(null as int64)', 'a.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'a.year', 'cast(null as int64)') }} as as_of_date_basis,
        a.unit,
        a.value,
        a.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_food_aid_conformed') }} a
    left join {{ ref('int_faostat_area_iso') }} iso
        on iso.area_code = cast(a.area_code as string)
    left join country_by_native_id g_nat
        on g_nat.native_id = cast(a.area_code as string)
    left join country_by_iso3 g_iso
        on upper(trim(g_iso.country_iso3)) = upper(trim(iso.country_iso3))
    left join {{ ref('dim_item') }} i
        on lower(i.item_name) = lower(a.item)
    left join {{ ref('dim_element') }} el
        on lower(el.element_name) = lower(a.element)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = a.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by humanitarian_key
    order by value desc nulls last
) = 1
