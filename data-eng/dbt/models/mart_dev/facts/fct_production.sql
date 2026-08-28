{{ config(

    materialized='table',

    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},

    cluster_by=['production_grain', 'data_level', 'country_iso3', 'source_key']

) }}



{# FAOSTAT country–year production (long-form). Yield FNID–season lives in fct_yield. #}



with {{ dim_country_by_native_id_cte() }},

{{ dim_country_by_iso3_cte() }},



base as (

    select

        to_hex(md5(

            'faostat|' || coalesce(cast(p.area_code as string), '') || '|' ||

            coalesce(cast(p.item_code as string), '') || '|' ||

            coalesce(cast(p.element_code as string), '') || '|' ||

            cast(p.year as string) || '|' || p.source_natural_key

        )) as production_key,

        coalesce(g_nat.geography_key, g_iso.geography_key) as geography_key,

        coalesce(g_nat.geo_level, g_iso.geo_level) as geo_level,

        {{ acf_country_iso3('coalesce(g_nat.country_iso3, g_iso.country_iso3)', 'iso.country_iso3') }} as country_iso3,

        cast(null as string) as season_key,

        pr.product_key,

        el.element_key,

        s.source_key,

        s.tier,

        {{ acf_row_data_level_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as data_level,

        {{ acf_geo_scope_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as geo_scope,

        {{ acf_place_scope_coalesce('g_nat', 'g_iso', 'iso.country_iso3', 'p.country_name') }} as place_scope,

        concat(

            'production_',

            lower(replace(coalesce(p.element, p.product_name, ''), ' ', '_')),

            '_',

            p.production_grain

        ) as metric,

        s.source_key as source_id,

        u.unit_key as production_unit_key,

        p.production_grain,

        p.element,

        p.year,

        cast(null as int64) as month,

        case

            when p.year is not null then format_date('%Y%m%d', date(p.year, 1, 1))

        end as date_key,

        {{ acf_as_of_date('cast(null as date)', 'p.year', 'cast(null as int64)', 'p.loaded_at') }} as as_of_date,

        {{ acf_as_of_date_basis('cast(null as date)', 'p.year', 'cast(null as int64)') }} as as_of_date_basis,

        'country_year' as grain_flag,

        case when lower(trim(p.element)) = 'area harvested' then p.value end as area_harvested,

        case when lower(trim(p.element)) = 'production' then p.value end as production_qty,

        case when lower(trim(p.element)) = 'yield' then p.value end as yield_value,

        case when lower(trim(p.element)) = 'area harvested' then p.unit end as area_unit,

        case when lower(trim(p.element)) = 'production' then p.unit end as production_unit,

        case when lower(trim(p.element)) = 'yield' then p.unit end as yield_unit,

        p.unit,

        p.value,

        p.source_natural_key,

        current_timestamp() as loaded_at

    from {{ ref('int_faostat_production_conformed') }} p

    left join {{ ref('int_faostat_area_iso') }} iso

        on iso.area_code = cast(p.area_code as string)

    left join country_by_native_id g_nat

        on g_nat.native_id = cast(p.area_code as string)

    left join country_by_iso3 g_iso

        on upper(trim(g_iso.country_iso3)) = upper(trim(iso.country_iso3))

    left join {{ ref('dim_product') }} pr

        on pr.item_code = cast(p.item_code as string)

       and lower(pr.product_name) = lower(p.product_name)

    left join {{ ref('dim_element') }} el

        on lower(el.element_name) = lower(p.element)

    left join {{ ref('dim_source') }} s

        on s.source_natural_key = p.source_natural_key

    left join {{ ref('dim_unit') }} u

        on lower(u.unit_code) = lower(p.unit)

       and u.unit_type = 'quantity'

)



select *

from base

where value is not null

