{{ config(materialized='table') }}

select
    f.*,
    g.geo_key
from {{ ref('int_fews_food_security_conformed') }} f
left join {{ ref('int_geography_conformed') }} g
    on g.geo_level = 'fnid'
   and g.fnid = f.fnid
