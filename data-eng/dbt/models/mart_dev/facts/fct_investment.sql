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
            coalesce(cast(inv.area_code as string), '') || '|' ||
            coalesce(cast(inv.donor_code as string), '') || '|' ||
            coalesce(cast(inv.item_code as string), '') || '|' ||
            coalesce(cast(inv.element_code as string), '') || '|' ||
            coalesce(inv.indicator, '') || '|' ||
            coalesce(inv.sex, '') || '|' ||
            coalesce(inv.institution, '') || '|' ||
            coalesce(cast(inv.year as string), '') || '|' ||
            coalesce(inv.source_natural_key, '')
        )) as investment_key,
        coalesce(g_nat.geography_key, g_iso.geography_key) as geography_key,
        coalesce(g_nat.geo_level, g_iso.geo_level) as geo_level,
        {{ acf_country_iso3('coalesce(g_nat.country_iso3, g_iso.country_iso3)', 'iso.country_iso3') }} as country_iso3,
        i.item_key,
        el.element_key,
        ind.indicator_key,
        sx.sex_key,
        org.organisation_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as data_level,
        {{ acf_geo_scope_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as geo_scope,
        {{ acf_place_scope_coalesce('g_nat', 'g_iso', 'iso.country_iso3', 'inv.country_name') }} as place_scope,
        lower(replace(coalesce(inv.indicator, inv.element, inv.item, ''), ' ', '_')) as metric,
        s.source_key as source_id,
        inv.donor,
        inv.purpose,
        inv.indicator,
        inv.institution,
        inv.sex,
        inv.year,
        format_date('%Y%m%d', date(inv.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 'inv.year', 'cast(null as int64)', 'inv.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'inv.year', 'cast(null as int64)') }} as as_of_date_basis,
        inv.unit,
        inv.value,
        inv.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_investment_asti_conformed') }} inv
    left join {{ ref('int_faostat_area_iso') }} iso
        on cast(iso.area_code as string) = cast(inv.area_code as string)
    left join country_by_native_id g_nat
        on g_nat.native_id = cast(inv.area_code as string)
    left join country_by_iso3 g_iso
        on upper(trim(g_iso.country_iso3)) = upper(trim(iso.country_iso3))
    left join {{ ref('dim_item') }} i
        on lower(i.item_name) = lower(inv.item)
    left join {{ ref('dim_element') }} el
        on lower(el.element_name) = lower(inv.element)
    left join {{ ref('dim_indicator') }} ind
        on lower(ind.indicator_name) = lower(inv.indicator)
    left join {{ ref('dim_sex') }} sx
        on sx.sex_key = case
            when lower(trim(inv.sex)) in ('male', 'm') then 'male'
            when lower(trim(inv.sex)) in ('female', 'f') then 'female'
            when lower(trim(inv.sex)) in ('total', 'both sexes', 'both') then 'total'
            else 'unknown'
        end
    left join {{ ref('dim_organisation') }} org
        on org.org_source = 'asti'
       and lower(org.legal_name) = lower(inv.institution)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = inv.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by investment_key
    order by value desc nulls last
) = 1
