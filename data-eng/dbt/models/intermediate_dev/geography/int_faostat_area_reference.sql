{{ config(materialized='table') }}

{# Canonical FAOSTAT area_code → ISO mapping. Priority: M49 → ref country name → alias. #}

with areas as (
    select * from {{ ref('int_faostat_areas_distinct') }}
),

m49 as (
    select
        {{ geo_faostat_m49_norm('m49_code') }} as m49_code,
        country_iso2,
        country_iso3,
        country_name as m49_country_name
    from {{ ref('ref_m49_country') }}
),

ref_country as (
    select
        lower(trim(country_name)) as country_name_norm,
        country_iso2,
        country_iso3,
        country_name
    from {{ ref('int_ref_country') }}
),

mapped as (
    select
        a.area_code,
        a.area_code_m49,
        a.country_name,
        coalesce(
            m49.country_iso3,
            rc.country_iso3,
            {{ geo_faostat_country_name_iso3('a.country_name') }}
        ) as country_iso3,
        case
            when m49.country_iso3 is not null then 'm49'
            when rc.country_iso3 is not null then 'ref_country_name'
            when {{ geo_faostat_country_name_iso3('a.country_name') }} is not null then 'alias'
        end as match_method
    from areas a
    left join m49
        on m49.m49_code = a.area_code_m49
    left join ref_country rc
        on rc.country_name_norm = lower(trim(a.country_name))
)

select
    m.area_code,
    m.area_code_m49,
    m.country_name,
    m.country_iso3,
    coalesce(rc2.country_iso2, m49b.country_iso2) as country_iso2,
    m.match_method,
    current_timestamp() as loaded_at
from mapped m
left join ref_country rc2
    on rc2.country_iso3 = m.country_iso3
left join m49 m49b
    on m49b.country_iso3 = m.country_iso3
