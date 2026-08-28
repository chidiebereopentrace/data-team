{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

select
    to_hex(md5(
        coalesce(h.country_iso3, '') || '|' ||
        cast(h.year as string) || '|' ||
        h.source_natural_key
    )) as hdi_key,
    g.geography_key,
    g.geo_level,
    {{ acf_country_iso3('g.country_iso3', 'h.country_iso3') }} as country_iso3,
    s.source_key,
    s.tier,
    {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
    {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
    {{ acf_place_scope('g') }} as place_scope,
    'hdi' as metric,
    s.source_key as source_id,
    h.year,
    format_date('%Y%m%d', date(h.year, 1, 1)) as date_key,
    {{ acf_as_of_date('cast(null as date)', 'h.year', 'cast(null as int64)', 'h.loaded_at') }} as as_of_date,
    {{ acf_as_of_date_basis('cast(null as date)', 'h.year', 'cast(null as int64)') }} as as_of_date_basis,
    h.hdi_value as value,
    cast(null as string) as unit,
    h.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('int_hdi_conformed') }} h
left join {{ ref('dim_geography') }} g
    on g.geography_key = h.geo_key
left join {{ ref('dim_source') }} s
    on s.source_natural_key = h.source_natural_key
