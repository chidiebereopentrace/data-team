{{ config(materialized='table') }}

with ref_country as (
    select
        country_iso2,
        country_iso3,
        country_name,
        in_africa_scope
    from {{ ref('int_ref_country') }}
),

cities as (
    select
        'city' as geo_level,
        trim(g.country) as country_name,
        upper(trim(g.iso2)) as country_iso2,
        upper(trim(g.iso3)) as country_iso3,
        trim(g.Admin_name) as admin_1_name,
        cast(null as string) as admin_2_name,
        trim(coalesce(g.city_ascii, g.city)) as city_name,
        cast(null as string) as fnid,
        safe_cast(g.lat as float64) as latitude,
        safe_cast(g.lng as float64) as longitude,
        safe_cast(g.population as int64) as population,
        trim(g.capital) as capital_status,
        cast(g.id as string) as native_id,
        'stg_geo' as source_natural_key
    from {{ source('staging_dev', 'stg_geo') }} g
    inner join ref_country rc
        on rc.country_iso2 = upper(trim(g.iso2))
       and rc.in_africa_scope
    where g.iso2 is not null
),

countries_from_cities as (
    select
        'country' as geo_level,
        country_name,
        country_iso2,
        country_iso3,
        cast(null as string) as admin_1_name,
        cast(null as string) as admin_2_name,
        cast(null as string) as city_name,
        cast(null as string) as fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        max(population) as population,
        cast(null as string) as capital_status,
        country_iso3 as native_id,
        'stg_geo_country' as source_natural_key
    from cities
    where country_iso3 is not null
    group by country_name, country_iso2, country_iso3
),

fews as (
    select distinct
        case
            when fnid is not null then 'fnid'
            when admin_2 is not null then 'admin2'
            when admin_1 is not null then 'admin1'
            else 'country'
        end as geo_level,
        f.country as country_name,
        upper(trim(f.country_code)) as country_iso2,
        rc.country_iso3,
        f.admin_1 as admin_1_name,
        f.admin_2 as admin_2_name,
        cast(null as string) as city_name,
        f.fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        cast(null as int64) as population,
        cast(null as string) as capital_status,
        coalesce(f.fnid, f.country_code) as native_id,
        f.source_natural_key
    from {{ ref('stg_fews_food_security') }} f
    left join ref_country rc
        on rc.country_iso2 = upper(trim(f.country_code))
    where f.country is not null
),

fews_prices as (
    select distinct
        case
            when admin_2 is not null then 'admin2'
            when admin_1 is not null then 'admin1'
            else 'country'
        end as geo_level,
        f.country as country_name,
        upper(trim(f.country_code)) as country_iso2,
        rc.country_iso3,
        f.admin_1 as admin_1_name,
        f.admin_2 as admin_2_name,
        cast(null as string) as city_name,
        cast(null as string) as fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        cast(null as int64) as population,
        cast(null as string) as capital_status,
        f.country_code as native_id,
        f.source_natural_key
    from {{ ref('stg_fews_market_prices') }} f
    left join ref_country rc
        on rc.country_iso2 = upper(trim(f.country_code))
    where f.country is not null
),

yield_geo as (
    select distinct
        case when y.fnid is not null then 'fnid' else 'country' end as geo_level,
        y.country as country_name,
        upper(trim(y.country_code)) as country_iso2,
        rc.country_iso3,
        y.admin_1 as admin_1_name,
        y.admin_2 as admin_2_name,
        cast(null as string) as city_name,
        y.fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        cast(null as int64) as population,
        cast(null as string) as capital_status,
        coalesce(y.fnid, y.country_code) as native_id,
        y.source_natural_key
    from {{ ref('stg_yield_raw_data') }} y
    left join ref_country rc
        on rc.country_iso2 = upper(trim(y.country_code))
    where y.country is not null
),

faostat_geo as (
    select distinct
        'country' as geo_level,
        iso.country_name,
        iso.country_iso2,
        iso.country_iso3,
        cast(null as string) as admin_1_name,
        cast(null as string) as admin_2_name,
        cast(null as string) as city_name,
        cast(null as string) as fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        cast(null as int64) as population,
        cast(null as string) as capital_status,
        iso.area_code as native_id,
        'faostat_area' as source_natural_key
    from {{ ref('int_faostat_area_iso') }} iso
    where iso.country_name is not null
      and iso.country_iso3 is not null
),

{# Country grain for FAOSTAT territories missing from city-derived countries; native_id = area_code. #}
countries_from_faostat_iso as (
    select
        'country' as geo_level,
        iso.country_name,
        iso.country_iso2,
        iso.country_iso3,
        cast(null as string) as admin_1_name,
        cast(null as string) as admin_2_name,
        cast(null as string) as city_name,
        cast(null as string) as fnid,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        cast(null as int64) as population,
        cast(null as string) as capital_status,
        iso.area_code as native_id,
        'faostat_area_iso_country' as source_natural_key
    from {{ ref('int_faostat_area_iso') }} iso
    where iso.country_iso3 is not null
      and not exists (
          select 1
          from countries_from_cities c
          where c.country_iso3 = iso.country_iso3
      )
    qualify row_number() over (
        partition by iso.area_code
        order by iso.country_name
    ) = 1
),

unioned as (
    select * from cities
    union all
    select * from countries_from_cities
    union all
    select * from countries_from_faostat_iso
    union all
    select * from fews
    union all
    select * from fews_prices
    union all
    select * from yield_geo
    union all
    select * from faostat_geo
),

africa as (
    select u.*
    from unioned u
    left join ref_country rc
        on rc.country_iso2 = u.country_iso2
    where u.country_iso2 is null
       or rc.in_africa_scope
),

iso2_to_iso3 as (
    select
        country_iso2,
        any_value(country_iso3) as country_iso3
    from countries_from_cities
    where country_iso2 is not null
      and country_iso3 is not null
    group by country_iso2
    union distinct
    select country_iso2, country_iso3
    from ref_country
    where country_iso2 is not null
      and country_iso3 is not null
),

keyed as (
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
        source_natural_key
    from africa
)

select
    k.geo_key,
    k.geo_level,
    k.country_name,
    k.country_iso2,
    coalesce(k.country_iso3, m.country_iso3) as country_iso3,
    k.admin_1_name,
    k.admin_2_name,
    k.city_name,
    k.fnid,
    k.latitude,
    k.longitude,
    k.population,
    k.capital_status,
    k.native_id,
    k.source_natural_key,
    current_timestamp() as loaded_at
from keyed k
left join iso2_to_iso3 m
    on m.country_iso2 = k.country_iso2
