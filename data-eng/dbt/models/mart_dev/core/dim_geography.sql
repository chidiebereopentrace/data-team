{{ config(
    materialized='table',
    cluster_by=['geo_level', 'country_iso3']
) }}

with base as (
    select
        geo_key as geography_key,
        geo_level,
        case
            when geo_level = 'country' then 'national'
            when geo_level in ('admin1', 'admin2', 'fnid') then 'sub_national'
            when geo_level = 'city' then 'community'
            else 'point'
        end as data_level,
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
        native_id
    from {{ ref('int_geography_conformed') }}
    qualify row_number() over (
        partition by geo_key
        order by
            case geo_level
                when 'city' then 1
                when 'fnid' then 2
                when 'admin2' then 3
                when 'admin1' then 4
                else 5
            end,
            population desc nulls last
    ) = 1
),

country_by_iso3 as (
    select *
    from base
    where geo_level = 'country'
      and country_iso3 is not null
    qualify row_number() over (
        partition by country_iso3
        order by population desc nulls last, geography_key
    ) = 1
),

country_by_iso2 as (
    select *
    from base
    where geo_level = 'country'
      and country_iso2 is not null
    qualify row_number() over (
        partition by country_iso2
        order by population desc nulls last, geography_key
    ) = 1
),

country_by_name as (
    select *
    from base
    where geo_level = 'country'
      and country_name is not null
    qualify row_number() over (
        partition by lower(trim(country_name))
        order by population desc nulls last, geography_key
    ) = 1
),

admin1_by_country as (
    select *
    from base
    where geo_level = 'admin1'
      and admin_1_name is not null
    qualify row_number() over (
        partition by
            coalesce(country_iso3, country_iso2, lower(trim(country_name))),
            lower(trim(admin_1_name))
        order by population desc nulls last, geography_key
    ) = 1
),

admin2_by_country as (
    select *
    from base
    where geo_level = 'admin2'
      and admin_2_name is not null
    qualify row_number() over (
        partition by
            coalesce(country_iso3, country_iso2, lower(trim(country_name))),
            lower(trim(coalesce(admin_1_name, ''))),
            lower(trim(admin_2_name))
        order by population desc nulls last, geography_key
    ) = 1
)

select
    b.geography_key,
    b.geo_level,
    b.data_level,
    b.country_name,
    b.country_iso2,
    b.country_iso3,
    coalesce(c3.geography_key, c2.geography_key, cn.geography_key) as country_key,
    b.admin_1_name,
    b.admin_2_name,
    b.city_name,
    b.fnid,
    case
        when b.geo_level = 'country' then cast(null as string)
        when b.geo_level = 'admin1' then coalesce(c3.geography_key, c2.geography_key, cn.geography_key)
        when b.geo_level = 'admin2' then coalesce(
            a1.geography_key,
            c3.geography_key, c2.geography_key, cn.geography_key
        )
        when b.geo_level in ('fnid', 'city') then coalesce(
            a2.geography_key,
            a1.geography_key,
            c3.geography_key, c2.geography_key, cn.geography_key
        )
        else coalesce(c3.geography_key, c2.geography_key, cn.geography_key)
    end as parent_geography_key,
    b.latitude,
    b.longitude,
    b.population,
    b.capital_status,
    b.native_id,
    current_timestamp() as loaded_at
from base b
left join country_by_iso3 c3
    on c3.country_iso3 = b.country_iso3
left join country_by_iso2 c2
    on c2.country_iso2 = b.country_iso2
   and b.country_iso3 is null
left join country_by_name cn
    on lower(trim(cn.country_name)) = lower(trim(b.country_name))
   and b.country_iso3 is null
   and b.country_iso2 is null
left join admin1_by_country a1
    on lower(trim(a1.admin_1_name)) = lower(trim(b.admin_1_name))
   and coalesce(a1.country_iso3, a1.country_iso2, lower(trim(a1.country_name)))
     = coalesce(b.country_iso3, b.country_iso2, lower(trim(b.country_name)))
left join admin2_by_country a2
    on lower(trim(a2.admin_2_name)) = lower(trim(b.admin_2_name))
   and lower(trim(coalesce(a2.admin_1_name, ''))) = lower(trim(coalesce(b.admin_1_name, '')))
   and coalesce(a2.country_iso3, a2.country_iso2, lower(trim(a2.country_name)))
     = coalesce(b.country_iso3, b.country_iso2, lower(trim(b.country_name)))
