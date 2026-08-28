{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'source_key']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(cast(a.sensor_id as string), '') || '|' ||
            coalesce(cast(a.observation_timestamp as string), '') || '|' ||
            coalesce(a.source_natural_key, '')
        )) as air_quality_key,
        a.geo_key as geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'a.country_iso3') }} as country_iso3,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        'pm2_5' as metric,
        s.source_key as source_id,
        a.sensor_id,
        a.latitude,
        a.longitude,
        safe_cast(a.observation_timestamp as timestamp) as observed_at,
        {{ acf_as_of_date(
            'date(safe_cast(a.observation_timestamp as timestamp))',
            'cast(null as int64)',
            'cast(null as int64)',
            'a.loaded_at'
        ) }} as as_of_date,
        {{ acf_as_of_date_basis(
            'date(safe_cast(a.observation_timestamp as timestamp))',
            'cast(null as int64)',
            'cast(null as int64)'
        ) }} as as_of_date_basis,
        a.pm2_5 as value,
        cast(null as string) as unit,
        a.pm2_5,
        a.pm10,
        a.humidity_pct,
        a.temperature_c,
        a.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_nakuru_air_conformed') }} a
    left join {{ ref('dim_geography') }} g
        on g.geography_key = a.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = a.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by air_quality_key
    order by observed_at desc nulls last
) = 1
