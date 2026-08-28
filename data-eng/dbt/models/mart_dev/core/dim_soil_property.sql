{{ config(materialized='table') }}

select
    to_hex(md5(lower(trim(soil_property)) || '|' || coalesce(depth, ''))) as soil_property_key,
    soil_property,
    depth,
    current_timestamp() as loaded_at
from (
    select soil_property, depth_band as depth from {{ ref('int_isric_soil_long') }}
    union all
    select soil_property, depth from {{ ref('int_isda_soil_with_geo') }}
)
where soil_property is not null
qualify row_number() over (
    partition by lower(trim(soil_property)), coalesce(depth, '')
    order by depth
) = 1
