{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

with country_by_name as (
    select *
    from {{ ref('dim_geography') }}
    where geo_level = 'country'
      and country_name is not null
    qualify row_number() over (
        partition by lower(trim(country_name))
        order by population desc nulls last, geography_key
    ) = 1
),

base as (
    select
        to_hex(md5(
            coalesce(i.source_natural_key, '') || '|' ||
            coalesce(i.household_id, '') || '|' ||
            coalesce(i.record_type, '')
        )) as insurance_key,
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3') }} as country_iso3,
        hh.household_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        concat('insurance_', lower(replace(coalesce(i.record_type, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        i.household_id,
        i.country,
        i.herd_size_category,
        i.insurance_start_year,
        {{ acf_as_of_date(
            'cast(null as date)',
            'safe_cast(i.insurance_start_year as int64)',
            'cast(null as int64)',
            'i.loaded_at'
        ) }} as as_of_date,
        {{ acf_as_of_date_basis(
            'cast(null as date)',
            'safe_cast(i.insurance_start_year as int64)',
            'cast(null as int64)'
        ) }} as as_of_date_basis,
        cast(null as float64) as value,
        cast(null as string) as unit,
        i.record_type,
        i.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_i4i_insurance_conformed') }} i
    left join country_by_name g
        on lower(trim(g.country_name)) = lower(trim(i.country))
    left join {{ ref('dim_household') }} hh
        on hh.household_id = i.household_id
       and hh.source_natural_key = i.source_natural_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = i.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by insurance_key
    order by insurance_start_year desc nulls last
) = 1
