{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(cast(e.area_code as string), '') || '|' ||
            coalesce(cast(e.item_code as string), '') || '|' ||
            coalesce(cast(e.element_code as string), '') || '|' ||
            coalesce(e.indicator, '') || '|' ||
            coalesce(cast(e.source_code as string), '') || '|' ||
            coalesce(e.sex, '') || '|' ||
            coalesce(cast(e.sex_code as string), '') || '|' ||
            cast(e.year as string) || '|' ||
            e.source_natural_key
        )) as employment_key,
        e.geo_key as geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'e.country_iso3') }} as country_iso3,
        i.item_key,
        el.element_key,
        ind.indicator_key,
        sx.sex_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope_coalesce('g', 'g', 'e.country_iso3', 'e.country_name') }} as place_scope,
        lower(replace(coalesce(e.indicator, e.element, e.item, ''), ' ', '_')) as metric,
        s.source_key as source_id,
        e.indicator,
        e.sex,
        e.year,
        format_date('%Y%m%d', date(e.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 'e.year', 'cast(null as int64)', 'e.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'e.year', 'cast(null as int64)') }} as as_of_date_basis,
        e.unit,
        e.value,
        e.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_employment_conformed') }} e
    left join {{ ref('dim_geography') }} g
        on g.geography_key = e.geo_key
    left join {{ ref('dim_item') }} i
        on lower(i.item_name) = lower(e.item)
    left join {{ ref('dim_element') }} el
        on lower(el.element_name) = lower(e.element)
    left join {{ ref('dim_indicator') }} ind
        on lower(ind.indicator_name) = lower(e.indicator)
    left join {{ ref('dim_sex') }} sx
        on sx.sex_key = case
            when lower(trim(e.sex)) in ('male', 'm') then 'male'
            when lower(trim(e.sex)) in ('female', 'f') then 'female'
            when lower(trim(e.sex)) in ('total', 'both sexes', 'both') then 'total'
            else 'unknown'
        end
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = e.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by employment_key
    order by value desc nulls last
) = 1
