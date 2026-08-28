{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

with country_geo as (
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
            coalesce(h.author_year, '') || '|' ||
            coalesce(h.study_site, '') || '|' ||
            coalesce(h.foodborne_hazard, '') || '|' ||
            coalesce(cast(h.publication_year as string), '') || '|' ||
            coalesce(h.source_natural_key, '')
        )) as food_hazard_key,
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3') }} as country_iso3,
        d.disease_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        concat('hazard_', lower(replace(coalesce(h.foodborne_hazard, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        h.author_year,
        h.publication_year,
        h.study_site,
        h.samples_type,
        h.foodborne_hazard,
        format_date('%Y%m%d', date(safe_cast(h.publication_year as int64), 1, 1)) as date_key,
        {{ acf_as_of_date(
            'cast(null as date)',
            'safe_cast(h.publication_year as int64)',
            'cast(null as int64)',
            'h.loaded_at'
        ) }} as as_of_date,
        {{ acf_as_of_date_basis(
            'cast(null as date)',
            'safe_cast(h.publication_year as int64)',
            'cast(null as int64)'
        ) }} as as_of_date_basis,
        h.total_samples,
        h.positive_samples,
        h.mean_cfu_per_g_log,
        h.country,
        h.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_food_hazards_conformed') }} h
    left join country_geo g
        on lower(trim(g.country_name)) = lower(trim(h.country))
    left join {{ ref('dim_disease') }} d
        on lower(d.disease_or_hazard) = lower(h.foodborne_hazard)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = h.source_natural_key
    where h.publication_year is not null
)

select *
from base
qualify row_number() over (
    partition by food_hazard_key
    order by total_samples desc nulls last
) = 1
