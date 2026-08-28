{{ config(materialized='table') }}

{# 0.1° lat/lon → country via geoBoundaries bbox (grid-scale). Per-point facts use ST_CONTAINS on geog. #}

with lat_cells as (
    select lat / 10.0 as lat_cell
    from unnest(generate_array(-350, 380, 1)) as lat
),

lon_cells as (
    select lon / 10.0 as lon_cell
    from unnest(generate_array(-250, 600, 1)) as lon
),

grid as (
    select
        lat_cell,
        lon_cell,
        lat_cell as latitude,
        lon_cell as longitude
    from lat_cells
    cross join lon_cells
    where {{ geo_africa_bbox_filter('lat_cell', 'lon_cell') }}
      and {{ geo_africa_soil_fringe_exclude('lat_cell', 'lon_cell') }}
),

matched as (
    select
        g.lat_cell,
        g.lon_cell,
        c.country_iso2,
        c.country_iso3,
        c.country_name,
        c.source_priority,
        (c.max_lat - c.min_lat) * (c.max_lng - c.min_lng) as country_area_proxy
    from grid g
    inner join {{ ref('int_admin0_africa') }} c
        on g.latitude between c.min_lat and c.max_lat
       and g.longitude between c.min_lng and c.max_lng
)

select
    lat_cell,
    lon_cell,
    country_iso2,
    country_iso3,
    country_name,
    current_timestamp() as loaded_at
from matched
qualify row_number() over (
    partition by cast(lat_cell as string), cast(lon_cell as string)
    order by source_priority asc nulls last, country_area_proxy asc nulls last, country_iso2
) = 1
