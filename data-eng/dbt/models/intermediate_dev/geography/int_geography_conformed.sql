{{ config(materialized='table') }}

{% set africa_iso2 = [
  'DZ','AO','BJ','BW','BF','BI','CM','CV','CF','TD','KM','CG','CD','CI','DJ',
  'EG','GQ','ER','SZ','ET','GA','GM','GH','GN','GW','KE','LS','LR','LY','MG',
  'MW','ML','MR','MU','MA','MZ','NA','NE','NG','RW','ST','SN','SC','SL','SO',
  'ZA','SS','SD','TZ','TG','TN','UG','ZM','ZW'
] %}

with cities as (
    select
        'city' as geo_level,
        trim(country) as country_name,
        upper(trim(iso2)) as country_iso2,
        upper(trim(iso3)) as country_iso3,
        trim(Admin_name) as admin_1_name,
        cast(null as string) as admin_2_name,
        trim(coalesce(city_ascii, city)) as city_name,
        cast(null as string) as fnid,
        safe_cast(lat as float64) as latitude,
        safe_cast(lng as float64) as longitude,
        safe_cast(population as int64) as population,
        trim(capital) as capital_status,
        cast(id as string) as native_id,
        'stg_geo' as source_natural_key
    from {{ source('staging_dev', 'stg_geo') }}
    where iso2 is not null
      and upper(trim(iso2)) in ({{ "'" ~ africa_iso2 | join("','") ~ "'" }})
),

fews as (
    select distinct
        case
            when fnid is not null then 'fnid'
            when admin_2 is not null then 'admin2'
            when admin_1 is not null then 'admin1'
            else 'country'
        end as geo_level,
        country as country_name,
        upper(trim(country_code)) as country_iso2,
        cast(null as string) as country_iso3,
        admin_1 as admin_1_name,
        admin_2 as admin_2_name,
        cast(null as string) as city_name,
        fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        cast(null as int64) as population,
        cast(null as string) as capital_status,
        coalesce(fnid, country_code) as native_id,
        source_natural_key
    from {{ ref('stg_fews_food_security') }}
    where country is not null
),

yield_geo as (
    select distinct
        case when fnid is not null then 'fnid' else 'country' end as geo_level,
        country as country_name,
        upper(trim(country_code)) as country_iso2,
        cast(null as string) as country_iso3,
        admin_1 as admin_1_name,
        admin_2 as admin_2_name,
        cast(null as string) as city_name,
        fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        cast(null as int64) as population,
        cast(null as string) as capital_status,
        coalesce(fnid, country_code) as native_id,
        source_natural_key
    from {{ ref('stg_yield_raw_data') }}
    where country is not null
),

faostat_geo as (
    select distinct
        'country' as geo_level,
        country_name,
        cast(null as string) as country_iso2,
        cast(null as string) as country_iso3,
        cast(null as string) as admin_1_name,
        cast(null as string) as admin_2_name,
        cast(null as string) as city_name,
        cast(null as string) as fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        cast(null as int64) as population,
        cast(null as string) as capital_status,
        cast(area_code as string) as native_id,
        source_natural_key
    from {{ ref('stg_faostat_production') }}
    where country_name is not null
),

unioned as (
    select * from cities
    union all
    select * from fews
    union all
    select * from yield_geo
    union all
    select * from faostat_geo
),

africa as (
    select *
    from unioned
    where country_iso2 in ({{ "'" ~ africa_iso2 | join("','") ~ "'" }})
       or country_iso2 is null
)

select
    to_hex(md5(
        coalesce(geo_level, '') || '|' ||
        coalesce(country_iso3, country_iso2, country_name, '') || '|' ||
        coalesce(admin_1_name, '') || '|' ||
        coalesce(admin_2_name, '') || '|' ||
        coalesce(city_name, '') || '|' ||
        coalesce(fnid, '')
    )) as geo_key,
    geo_level,
    country_name,
    country_iso2,
    country_iso3,
    admin_1_name,
    admin_2_name,
    city_name,
    fnid,
    latitude,
    longitude,
    population,
    capital_status,
    native_id,
    source_natural_key,
    current_timestamp() as loaded_at
from africa
