{{ config(materialized='table') }}

with iso3_geo as (
    select
        upper(trim(country_iso3)) as country_iso3,
        geo_key
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'country'
      and country_iso3 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso3))
        order by population desc nulls last, geo_key
    ) = 1
)

select
    p.country_name,
    upper(trim(p.country_code)) as country_iso3,
    p.observation_year as year,
    p.gdp_per_capita_ppp,
    g.geo_key,
    p.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_africa_gdp_ppp') }} p
left join iso3_geo g
    on g.country_iso3 = upper(trim(p.country_code))
where p.gdp_per_capita_ppp is not null
  and p.observation_year is not null
