{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(g.geography_key, '') || '|' ||
        coalesce(s.soil_property, '') || '|' ||
        coalesce(s.depth, '') || '|' ||
        coalesce(s.source_key, '')
    )) as agg_soil_admin_key,
    g.geography_key,
    g.country_name,
    g.admin_1_name,
    s.soil_property,
    s.depth,
    s.source_key,
    avg(s.value) as value_avg,
    count(*) as point_count,
    current_timestamp() as loaded_at
from {{ ref('fct_soil_health') }} s
left join {{ ref('dim_geography') }} g
    on g.geography_key = s.geography_key
where s.geography_key is not null
group by
    g.geography_key,
    g.country_name,
    g.admin_1_name,
    s.soil_property,
    s.depth,
    s.source_key
