{{ config(materialized='table') }}

{# Canonical country reference: stg_geo cities + M49 seed territories + in_africa_scope. #}

with from_cities as (
    select
        trim(country) as country_name,
        upper(trim(iso2)) as country_iso2,
        upper(trim(iso3)) as country_iso3,
        max(safe_cast(population as int64)) as population
    from {{ source('staging_dev', 'stg_geo') }}
    where iso2 is not null
      and iso3 is not null
      and country is not null
    group by 1, 2, 3
),

from_m49 as (
    select
        country_name,
        country_iso2,
        country_iso3,
        cast(null as int64) as population,
        in_africa_scope,
        lpad(m49_code, 3, '0') as m49_code
    from {{ ref('ref_m49_country') }}
),

merged as (
    select
        coalesce(c.country_name, m.country_name) as country_name,
        coalesce(c.country_iso2, m.country_iso2) as country_iso2,
        coalesce(c.country_iso3, m.country_iso3) as country_iso3,
        coalesce(c.population, m.population) as population,
        coalesce(m.in_africa_scope, false) as in_africa_scope,
        m.m49_code
    from from_cities c
    full outer join from_m49 m
        on m.country_iso3 = c.country_iso3
)

select
    country_name,
    country_iso2,
    country_iso3,
    population,
    in_africa_scope,
    m49_code,
    current_timestamp() as loaded_at
from merged
where country_iso3 is not null
qualify row_number() over (
    partition by country_iso3
    order by population desc nulls last, country_name
) = 1
