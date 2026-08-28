{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(b.gbif_id, '') || '|' ||
            coalesce(b.source_natural_key, '')
        )) as biodiversity_key,
        b.geo_key as geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'b.country_iso3') }} as country_iso3,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        'species_occurrence' as metric,
        s.source_key as source_id,
        b.gbif_id,
        b.scientific_name,
        b.country_code,
        b.latitude,
        b.longitude,
        b.individual_count,
        b.year,
        {{ acf_as_of_date(
            'safe_cast(b.event_date as date)',
            'safe_cast(b.year as int64)',
            'safe_cast(b.month as int64)',
            'b.loaded_at'
        ) }} as as_of_date,
        {{ acf_as_of_date_basis(
            'safe_cast(b.event_date as date)',
            'safe_cast(b.year as int64)',
            'safe_cast(b.month as int64)'
        ) }} as as_of_date_basis,
        cast(b.individual_count as float64) as value,
        cast(null as string) as unit,
        b.rich_all,
        b.rar_all,
        b.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_biodiversity_conformed') }} b
    left join {{ ref('dim_geography') }} g
        on g.geography_key = b.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = b.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by biodiversity_key
    order by individual_count desc nulls last, year desc nulls last
) = 1
