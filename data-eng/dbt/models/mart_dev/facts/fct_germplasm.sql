{{ config(
    materialized='table',
    cluster_by=['source_key', 'country_iso3']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(cast(g.germplasm_id as string), '') || '|' ||
            coalesce(g.source_natural_key, '')
        )) as germplasm_key,
        g.geo_key as geography_key,
        geo.geo_level,
        {{ acf_country_iso3('geo.country_iso3', 'g.country_iso3') }} as country_iso3,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('geo.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('geo.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('geo') }} as place_scope,
        'germplasm_accession' as metric,
        s.source_key as source_id,
        {{ acf_as_of_date('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)', 'g.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
        cast(null as float64) as value,
        cast(null as string) as unit,
        g.germplasm_id,
        g.taxon,
        g.latitude,
        g.longitude,
        g.geometry_wkt,
        g.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_germplasm_conformed') }} g
    left join {{ ref('dim_geography') }} geo
        on geo.geography_key = g.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = g.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by germplasm_key
    order by taxon
) = 1
