{{ config(materialized='table') }}

select
    y.fnid,
    y.country,
    y.country_code,
    y.admin_1,
    y.admin_2,
    y.product,
    y.season_name,
    s.season_key,
    s.season_country_key,
    y.planting_year,
    y.planting_month,
    y.harvest_year,
    y.harvest_month,
    y.crop_production_system,
    y.qc_flag,
    y.area,
    y.production,
    y.yield,
    coalesce(g_fnid.geo_key, g_admin.geo_key) as geo_key,
    coalesce(g_fnid.country_iso3, g_admin.country_iso3) as country_iso3,
    y.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('int_yield_raw_enriched') }} y
left join {{ ref('int_season_mapped') }} s
    on lower(trim(y.season_name)) = s.season_name_norm
   and y.country = s.country
left join {{ ref('int_geography_conformed') }} g_fnid
    on g_fnid.geo_level = 'fnid'
   and g_fnid.fnid = y.fnid
left join {{ ref('int_geography_conformed') }} g_admin
    on y.fnid is null
   and (
        (y.admin_1 is not null and g_admin.geo_level = 'admin1')
        or (y.admin_1 is null and g_admin.geo_level = 'country')
   )
   and upper(trim(g_admin.country_iso2)) = upper(trim(y.country_code))
   and coalesce(g_admin.admin_1_name, '') = coalesce(y.admin_1, '')
