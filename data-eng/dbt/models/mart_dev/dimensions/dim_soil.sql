{{ config(materialized='table') }}

-- Gold: soil depth-layer dimension for soil health fact joins.

with depth_layers as (
    select '0-5'   as depth_layer, 0  as depth_min_cm, 5   as depth_max_cm, 'Surface layer (0–5 cm)'    as description
    union all select '5-15',       5,                  15,                  'Topsoil (5–15 cm)'
    union all select '15-30',      15,                 30,                  'Subsoil upper (15–30 cm)'
    union all select '30-60',      30,                 60,                  'Subsoil lower (30–60 cm)'
    union all select '60-100',     60,                 100,                 'Deep soil (60–100 cm)'
    union all select '100-200',    100,                200,                 'Very deep soil (100–200 cm)'
)

select
    to_hex(md5(coalesce(depth_layer, '')))         as soil_key,
    depth_layer,
    depth_min_cm,
    depth_max_cm,
    description

from depth_layers
