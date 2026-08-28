{{ config(materialized='table') }}

with {{ geo_africa_cities_cte('africa_cities') }},
{{ geo_city_by_latlon_cte('city_by_latlon') }},
{{ geo_latlon_grid_cte('latlon_grid') }},
{{ geo_country_by_iso2_cte('iso2_geo') }},

src as (
    select
        objectid,
        protected_area_name,
        feature_count,
        area_protected,
        geometry_wkt,
        source_natural_key,
        {{ geo_centroid_latitude('geometry_wkt') }} as latitude,
        {{ geo_centroid_longitude('geometry_wkt') }} as longitude,
        current_timestamp() as loaded_at
    from {{ ref('stg_protected_areas') }}
),

in_bbox as (
    select *
    from src
    where latitude is null
       or longitude is null
       or {{ geo_africa_bbox_filter('latitude', 'longitude') }}
),

with_exact as (
    select
        s.*,
        g_exact.geo_key as exact_geo_key,
        g_exact.country_iso2 as exact_iso2,
        g_exact.country_iso3 as exact_iso3
    from in_bbox s
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
    from with_grid e
    left join africa_cities c
        on e.exact_geo_key is null
       and e.latitude is not null
       and e.longitude is not null
       and {{ geo_nearest_city_bbox_join('e', 'c', 5) }}
    qualify e.exact_geo_key is not null
        or e.latitude is null
        or e.longitude is null
        or row_number() over (
            partition by
                coalesce(cast(e.objectid as string), ''),
                coalesce(e.protected_area_name, ''),
                coalesce(e.source_natural_key, '')
            order by {{ geo_st_distance_m('e', 'c') }} nulls last, c.geo_key
        ) = 1
)

select
    n.objectid,
    n.protected_area_name,
    n.feature_count,
    n.area_protected,
    n.geometry_wkt,
    n.latitude,
    n.longitude,
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
    n.source_natural_key,
    n.loaded_at
from with_nearest n
left join iso2_geo g_country
    on g_country.country_iso2 = coalesce(
        n.exact_iso2,
        case when n.nearest_dist_m <= 100000 then n.nearest_iso2 end,
        n.grid_iso2,
        n.nearest_iso2
    )
