{{ config(
    materialized='table',
    cluster_by=['source_key', 'soil_property', 'data_level']
) }}

with base as (
    select
        to_hex(md5(
            'isric|' || cast(i.latitude as string) || '|' ||
            cast(i.longitude as string) || '|' ||
            coalesce(i.soil_property, '') || '|' ||
            coalesce(i.depth_band, '') || '|' ||
            i.source_natural_key
        )) as soil_key,
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'i.country_iso3') }} as country_iso3,
        sp.soil_property_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        concat('soil_', lower(replace(coalesce(i.soil_property, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        i.latitude,
        i.longitude,
        i.soil_property,
        i.depth_band as depth,
        i.value,
        {{ acf_as_of_date('i.fetched_date', 'cast(null as int64)', 'cast(null as int64)', 'i.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('i.fetched_date', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
        cast(null as string) as unit,
        i.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_isric_soil_with_geo') }} i
    left join {{ ref('dim_geography') }} g
        on g.geography_key = i.geo_key
    left join {{ ref('dim_soil_property') }} sp
        on lower(sp.soil_property) = lower(i.soil_property)
       and coalesce(sp.depth, '') = coalesce(i.depth_band, '')
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = i.source_natural_key

    union all

    select
        to_hex(md5(
            'isda|' || cast(d.latitude as string) || '|' ||
            cast(d.longitude as string) || '|' ||
            coalesce(d.soil_property, '') || '|' ||
            coalesce(d.depth, '') || '|' ||
            d.source_natural_key
        )),
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'd.country_iso3') }} as country_iso3,
        sp.soil_property_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        concat('soil_', lower(replace(coalesce(d.soil_property, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        d.latitude,
        d.longitude,
        d.soil_property,
        d.depth,
        d.value,
        {{ acf_as_of_date('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)', 'd.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
        cast(null as string) as unit,
        d.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_isda_soil_with_geo') }} d
    left join {{ ref('dim_geography') }} g
        on g.geography_key = d.geo_key
    left join {{ ref('dim_soil_property') }} sp
        on lower(sp.soil_property) = lower(d.soil_property)
       and coalesce(sp.depth, '') = coalesce(d.depth, '')
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = d.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by soil_key
    order by value desc nulls last
) = 1
