{{ config(materialized='table') }}

with {{ geo_africa_cities_cte('africa_cities') }},
{{ geo_city_by_latlon_cte('city_by_latlon') }},
{{ geo_latlon_grid_cte('latlon_grid') }},
{{ geo_country_by_iso2_cte('iso2_geo') }},

src as (
    select * from {{ ref('int_isric_soil_long') }}
),

with_exact as (
    select
        s.*,
        g_exact.geo_key as exact_geo_key,
        g_exact.country_iso2 as exact_iso2,
        g_exact.country_iso3 as exact_iso3,
        g_exact.country_name as exact_country_name
    from src s
    left join city_by_latlon g_exact
        on {{ geo_city_latlon_join('s', 'g_exact') }}
),

with_grid as (
    select
        e.*,
        g_grid.country_iso2 as grid_iso2,
        g_grid.country_iso3 as grid_iso3,
        g_grid.country_name as grid_country_name
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
        c.country_name as nearest_country_name,
        {{ geo_st_distance_m('e', 'c') }} as nearest_dist_m
    from with_grid e
    left join africa_cities c
        on e.exact_geo_key is null
       and {{ geo_nearest_city_bbox_join('e', 'c', 5) }}
    qualify e.exact_geo_key is not null
        or row_number() over (
            partition by
                cast(e.latitude as string),
                cast(e.longitude as string),
                coalesce(cast(e.fetched_date as string), ''),
                coalesce(e.soil_property, ''),
                coalesce(e.depth_band, ''),
                coalesce(e.source_natural_key, '')
            order by {{ geo_st_distance_m('e', 'c') }} nulls last, c.geo_key
        ) = 1
)

select *
from (
    select
        n.latitude,
        n.longitude,
        n.fetched_date,
        n.soil_property,
        n.depth_band,
        n.value,
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
        coalesce(
            n.exact_country_name,
            case when n.nearest_dist_m <= 100000 then n.nearest_country_name end,
            n.grid_country_name,
            g_country.country_name,
            n.nearest_country_name
        ) as country_name,
        current_timestamp() as loaded_at
    from with_nearest n
    left join iso2_geo g_country
        on g_country.country_iso2 = coalesce(
            n.exact_iso2,
            case when n.nearest_dist_m <= 100000 then n.nearest_iso2 end,
            n.grid_iso2,
            n.nearest_iso2
        )
) resolved
where country_iso2 is null
   or {{ geo_africa_iso2_in('country_iso2') }}
