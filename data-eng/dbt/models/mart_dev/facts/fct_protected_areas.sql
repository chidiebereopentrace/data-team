{{ config(
    materialized='table',
    cluster_by=['source_key', 'country_iso3']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(p.protected_area_name, '') || '|' ||
            coalesce(cast(p.objectid as string), '') || '|' ||
            coalesce(p.source_natural_key, '')
        )) as protected_area_key,
        p.geo_key as geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'p.country_iso3') }} as country_iso3,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope_coalesce('g', 'g', 'p.country_iso3', 'p.protected_area_name') }} as place_scope,
        'protected_area_extent' as metric,
        s.source_key as source_id,
        {{ acf_as_of_date('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)', 'p.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
        p.area_protected as value,
        cast(null as string) as unit,
        p.protected_area_name,
        p.latitude,
        p.longitude,
        p.geometry_wkt,
        p.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_protected_areas') }} p
    left join {{ ref('dim_geography') }} g
        on g.geography_key = p.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = p.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by protected_area_key
    order by value desc nulls last
) = 1
