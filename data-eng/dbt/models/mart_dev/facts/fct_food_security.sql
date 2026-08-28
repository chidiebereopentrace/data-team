{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['measure_type', 'data_level', 'country_iso3', 'source_key']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(f.fnid, '') || '|' ||
            coalesce(f.measure_type, '') || '|' ||
            coalesce(f.phase_code, '') || '|' ||
            coalesce(f.phase_name, '') || '|' ||
            coalesce(f.classification_scale, '') || '|' ||
            coalesce(f.scenario_name, '') || '|' ||
            cast(f.year as string) || '|' ||
            cast(f.month as string) || '|' ||
            coalesce(cast(f.is_allowing_for_assistance as string), '') || '|' ||
            coalesce(cast(f.value as string), '') || '|' ||
            coalesce(cast(f.low_value as string), '') || '|' ||
            coalesce(cast(f.high_value as string), '') || '|' ||
            f.source_natural_key
        )) as food_security_key,
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'f.country_iso3') }} as country_iso3,
        c.classification_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        concat(lower(f.measure_type), '_', lower(coalesce(f.phase_code, ''))) as metric,
        s.source_key as source_id,
        f.measure_type,
        f.scenario_name,
        f.year,
        f.month,
        case
            when f.year is not null and f.month is not null
                then format_date('%Y%m%d', date(f.year, f.month, 1))
        end as date_key,
        {{ acf_as_of_date('cast(null as date)', 'f.year', 'f.month', 'f.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'f.year', 'f.month') }} as as_of_date_basis,
        f.value,
        case
            when f.measure_type = 'classification' then 'IPC phase'
            when f.measure_type = 'population' then 'persons'
        end as unit,
        f.low_value,
        f.high_value,
        f.pct_phase3,
        f.pct_phase4,
        f.pct_phase5,
        f.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_fews_food_security_with_geo') }} f
    left join {{ ref('dim_geography') }} g
        on g.geography_key = f.geo_key
    left join {{ ref('dim_classification') }} c
        on c.phase_code = f.phase_code
       and coalesce(c.phase_name, '') = coalesce(f.phase_name, '')
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = f.source_natural_key
    where f.year is not null
      and f.month between 1 and 12
)

select *
from base
qualify row_number() over (
    partition by food_security_key
    order by value desc nulls last
) = 1
