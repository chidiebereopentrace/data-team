{{ config(materialized='table') }}

with {{ geo_africa_cities_cte('africa_cities') }},
{{ geo_city_by_latlon_cte('city_by_latlon') }},
{{ geo_latlon_grid_cte('latlon_grid') }},
{{ geo_country_by_iso2_cte('iso2_geo') }},

src as (
    select *
    from {{ ref('stg_nakuru_air_quality') }}
    where latitude between -35 and 38
      and longitude between -25 and 60
),

with_exact as (
    select
        s.*,
        g_exact.geo_key as exact_geo_key,
        g_exact.country_iso2 as exact_iso2,
        g_exact.country_iso3 as exact_iso3
    from src s
    left join city_by_latlon g_exact
        on {{ geo_city_latlon_join('s', 'g_exact') }}
),

with_grid as (
    select
        e.*,
        g_grid.country_iso2 as grid_iso2,
        g_grid.country_iso3 as grid_iso3
    from with_exact e
    left join latlon_grid g_grid
        on {{ geo_latlon_grid_join('e', 'g_grid') }}
),

with_nearest as (
    select
        e.*,
        c.geo_key as nearest_geo_key,
        c.country_iso2 as nearest_iso2,
        c.country_iso3 as nearest_iso3,
        {{ geo_st_distance_m('e', 'c') }} as nearest_dist_m
    from with_grid e
    left join africa_cities c
        on e.exact_geo_key is null
       and {{ geo_nearest_city_bbox_join('e', 'c') }}
    qualify e.exact_geo_key is not null
        or row_number() over (
            partition by
                coalesce(cast(e.sensor_id as string), ''),
                coalesce(cast(e.observation_timestamp as string), ''),
                coalesce(e.source_natural_key, '')
            order by {{ geo_st_distance_m('e', 'c') }} nulls last, c.geo_key
        ) = 1
)

select
    n.sensor_id,
    n.sensor_type,
    n.location,
    n.latitude,
    n.longitude,
    n.observation_timestamp,
    n.pm10,
    n.pm2_5,
    n.humidity_pct,
    n.temperature_c,
    n.source_archive,
    n.source_natural_key,
    {{ geo_resolve_point_geo_key(
        'n.exact_geo_key',
        'n.nearest_geo_key',
        'n.nearest_dist_m',
        'g_country.geo_key'
    ) }} as geo_key,
    coalesce(
        n.exact_iso2,
        case when n.nearest_dist_m <= 100000 then n.nearest_iso2 end,
        n.grid_iso2,
        g_country.country_iso2,
        n.nearest_iso2
    ) as country_iso2,
    coalesce(
        n.exact_iso3,
        case when n.nearest_dist_m <= 100000 then n.nearest_iso3 end,
        n.grid_iso3,
        g_country.country_iso3,
        n.nearest_iso3
    ) as country_iso3,
    current_timestamp() as loaded_at
from with_nearest n
left join iso2_geo g_country
    on g_country.country_iso2 = coalesce(
        n.exact_iso2,
        case when n.nearest_dist_m <= 100000 then n.nearest_iso2 end,
        n.grid_iso2,
        n.nearest_iso2
    )
