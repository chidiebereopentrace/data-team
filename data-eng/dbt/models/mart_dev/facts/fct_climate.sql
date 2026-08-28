{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['climate_grain', 'data_level', 'source_key']
) }}

with base as (
    select
        to_hex(md5(
            'nasa|' || coalesce(cast(n.observation_time as string), '') || '|' ||
            coalesce(cast(n.latitude as string), '') || '|' ||
            coalesce(cast(n.longitude as string), '') || '|' ||
            'par_solar_at_noon' || '|' ||
            n.source_natural_key
        )) as climate_key,
        n.geo_key as geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'n.country_iso3') }} as country_iso3,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        'par_solar_at_noon' as metric,
        s.source_key as source_id,
        cast(null as string) as indicator_key,
        'point_obs' as climate_grain,
        n.latitude,
        n.longitude,
        {{ acf_as_of_date('date(n.observation_time)', 'cast(null as int64)', 'cast(null as int64)', 'n.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('date(n.observation_time)', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
        format_date('%Y%m%d', date(n.observation_time)) as date_key,
        n.par_solar_at_noon as value,
        cast(null as string) as unit,
        n.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_nasa_power_conformed') }} n
    left join {{ ref('dim_geography') }} g
        on g.geography_key = n.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = n.source_natural_key
    where n.par_solar_at_noon is not null
      and n.observation_time is not null

    union all

    select
        to_hex(md5(
            'nasa|' || coalesce(cast(n.observation_time as string), '') || '|' ||
            coalesce(cast(n.latitude as string), '') || '|' ||
            coalesce(cast(n.longitude as string), '') || '|' ||
            'shortwave_irradiance_at_noon' || '|' ||
            n.source_natural_key
        )),
        n.geo_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'n.country_iso3') }},
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }},
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }},
        {{ acf_place_scope('g') }},
        'shortwave_irradiance_at_noon' as metric,
        s.source_key as source_id,
        cast(null as string) as indicator_key,
        'point_obs' as climate_grain,
        n.latitude,
        n.longitude,
        {{ acf_as_of_date('date(n.observation_time)', 'cast(null as int64)', 'cast(null as int64)', 'n.loaded_at') }},
        {{ acf_as_of_date_basis('date(n.observation_time)', 'cast(null as int64)', 'cast(null as int64)') }},
        format_date('%Y%m%d', date(n.observation_time)) as date_key,
        n.shortwave_irradiance_at_noon as value,
        cast(null as string) as unit,
        n.source_natural_key,
        current_timestamp()
    from {{ ref('int_nasa_power_conformed') }} n
    left join {{ ref('dim_geography') }} g
        on g.geography_key = n.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = n.source_natural_key
    where n.shortwave_irradiance_at_noon is not null
      and n.observation_time is not null

    union all

    select
        to_hex(md5(
            'era5|' || coalesce(cast(e.valid_time as string), '') || '|' ||
            coalesce(cast(e.latitude as string), '') || '|' ||
            coalesce(cast(e.longitude as string), '') || '|' ||
            'temperature_2m' || '|' ||
            e.source_natural_key
        )),
        e.geo_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'e.country_iso3') }},
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }},
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }},
        {{ acf_place_scope('g') }},
        'temperature_2m' as metric,
        s.source_key as source_id,
        cast(null as string) as indicator_key,
        'point_obs' as climate_grain,
        e.latitude,
        e.longitude,
        {{ acf_as_of_date('date(e.valid_time)', 'cast(null as int64)', 'cast(null as int64)', 'e.loaded_at') }},
        {{ acf_as_of_date_basis('date(e.valid_time)', 'cast(null as int64)', 'cast(null as int64)') }},
        format_date('%Y%m%d', date(e.valid_time)) as date_key,
        e.temperature_2m as value,
        cast(null as string) as unit,
        e.source_natural_key,
        current_timestamp()
    from {{ ref('int_era5_conformed') }} e
    left join {{ ref('dim_geography') }} g
        on g.geography_key = e.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = e.source_natural_key
    where e.valid_time is not null

    union all

    select
        to_hex(md5(
            'cw|' || coalesce(c.country_iso2, '') || '|' ||
            coalesce(c.indicator, '') || '|' || coalesce(c.scenario, '') || '|' ||
            cast(c.year as string) || '|' || c.source_natural_key
        )),
        c.geo_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'c.country_iso3') }},
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }},
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }},
        {{ acf_place_scope_coalesce('g', 'g', 'c.country_iso3', 'c.country_name') }},
        lower(replace(coalesce(c.indicator, ''), ' ', '_')) as metric,
        s.source_key as source_id,
        ind.indicator_key,
        'country_model' as climate_grain,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        {{ acf_as_of_date('cast(null as date)', 'c.year', 'cast(null as int64)', 'c.loaded_at') }},
        {{ acf_as_of_date_basis('cast(null as date)', 'c.year', 'cast(null as int64)') }},
        format_date('%Y%m%d', date(c.year, 1, 1)) as date_key,
        c.value,
        cast(null as string) as unit,
        c.source_natural_key,
        current_timestamp()
    from {{ ref('int_climatewatch_conformed') }} c
    left join {{ ref('dim_geography') }} g
        on g.geography_key = c.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = c.source_natural_key
    left join {{ ref('dim_indicator') }} ind
        on lower(ind.indicator_name) = lower(c.indicator)
    where c.year is not null
)

select *
from base
qualify row_number() over (
    partition by climate_key
    order by value desc nulls last
) = 1
