{{ config(materialized='table') }}

with {{ geo_africa_cities_cte('africa_cities') }},
{{ geo_city_by_latlon_cte('city_by_latlon') }},
{{ geo_latlon_grid_cte('latlon_grid') }},
{{ geo_country_by_iso2_cte('iso2_geo') }},
{{ geo_admin0_africa_cte('admin0_africa') }},

src as (
    select *
    from {{ ref('stg_nasa_power') }}
    where (
        latitude is null
        or longitude is null
        or (
            longitude between -25 and 60
            and latitude between -35 and 38
        )
    )
),

with_exact as (
    select
        s.*,
        g_exact.geo_key as exact_geo_key,
        g_exact.country_iso2 as exact_iso2,
        g_exact.country_iso3 as exact_iso3
    from src s
    left join city_by_latlon g_exact
        on s.latitude is not null
       and s.longitude is not null
       and {{ geo_city_latlon_join('s', 'g_exact') }}
),

with_grid as (
    select
        e.*,
        g_grid.country_iso2 as grid_iso2,
        g_grid.country_iso3 as grid_iso3
    from with_exact e
    left join latlon_grid g_grid
        on e.latitude is not null
       and e.longitude is not null
       and {{ geo_latlon_grid_join('e', 'g_grid') }}
),

with_admin0 as (
    select
        e.*,
        a.country_iso2 as admin0_iso2,
        a.country_iso3 as admin0_iso3
    from with_grid e
    left join admin0_africa a
        on e.latitude is not null
       and e.longitude is not null
       and {{ geo_admin0_bbox_prefilter('e', 'a') }}
    qualify e.latitude is null
        or e.longitude is null
        or a.country_iso2 is null
        or row_number() over (
            partition by
                coalesce(cast(e.latitude as string), ''),
                coalesce(cast(e.longitude as string), ''),
                coalesce(cast(e.observation_time as string), ''),
                coalesce(e.source_natural_key, '')
            order by a.source_priority asc nulls last, (a.max_lat - a.min_lat) * (a.max_lng - a.min_lng) asc nulls last, a.country_iso2
        ) = 1
),

with_nearest as (
    select
        e.*,
        c.geo_key as nearest_geo_key,
        c.country_iso2 as nearest_iso2,
        c.country_iso3 as nearest_iso3,
        case
            when e.latitude is not null and e.longitude is not null
                then {{ geo_st_distance_m('e', 'c') }}
        end as nearest_dist_m
    from with_admin0 e
    left join africa_cities c
        on e.exact_geo_key is null
       and e.latitude is not null
       and e.longitude is not null
       and {{ geo_nearest_city_bbox_join('e', 'c') }}
    qualify e.exact_geo_key is not null
        or e.latitude is null
        or e.longitude is null
        or row_number() over (
            partition by
                coalesce(cast(e.latitude as string), ''),
                coalesce(cast(e.longitude as string), ''),
                coalesce(cast(e.observation_time as string), ''),
                coalesce(e.source_natural_key, '')
            order by {{ geo_st_distance_m('e', 'c') }} nulls last, c.geo_key
        ) = 1
)

select
    n.country_code,
    n.country_name,
    n.admin_region,
    n.latitude,
    n.longitude,
    n.elevation_meters,
    n.par_solar_at_noon,
    n.shortwave_irradiance_at_noon,
    n.uva_radiation_at_noon,
    n.uvb_radiation_at_noon,
    n.observation_time,
    coalesce(
        {{ geo_resolve_point_geo_key(
            'n.exact_geo_key',
            'n.nearest_geo_key',
            'n.nearest_dist_m',
            'g_nearest_country.geo_key'
        ) }},
        g_code.geo_key
    ) as geo_key,
    coalesce(
        n.exact_iso2,
        case when n.nearest_dist_m <= 100000 then n.nearest_iso2 end,
        n.grid_iso2,
        n.admin0_iso2,
        g_nearest_country.country_iso2,
        n.nearest_iso2,
        g_code.country_iso2
    ) as country_iso2,
    coalesce(
        n.exact_iso3,
        case when n.nearest_dist_m <= 100000 then n.nearest_iso3 end,
        n.grid_iso3,
        n.admin0_iso3,
        g_nearest_country.country_iso3,
        n.nearest_iso3,
        g_code.country_iso3
    ) as country_iso3,
    n.source_natural_key,
    current_timestamp() as loaded_at
from with_nearest n
left join iso2_geo g_nearest_country
    on g_nearest_country.country_iso2 = {{ geo_iso2_for_country_dim_join(
        'n.exact_iso2',
        'n.nearest_dist_m',
        'n.nearest_iso2',
        'n.grid_iso2',
        'n.nearest_iso2',
        'n.admin0_iso2'
    ) }}
left join iso2_geo g_code
    on n.exact_geo_key is null
   and n.nearest_geo_key is null
   and upper(trim(n.country_code)) = g_code.country_iso2
