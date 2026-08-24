{{ config(materialized='table') }}

with country_geo as (
    select
        lower(trim(country_name)) as country_name_norm,
        geo_key
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'country'
      and country_name is not null
    qualify row_number() over (
        partition by lower(trim(country_name))
        order by geo_key
    ) = 1
)

select
    c.country_name,
    upper(trim(c.country_code)) as country_iso2,
    c.model,
    c.scenario,
    c.category,
    c.subcategory,
    c.indicator,
    c.unit,
    c.year,
    c.value,
    g.geo_key,
    c.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_climatewatch_health') }} c
left join country_geo g
    on g.country_name_norm = lower(trim(c.country_name))
