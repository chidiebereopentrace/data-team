{{ config(materialized='table') }}

with {{ geo_country_by_iso2_cte('iso2_geo') }},
{{ geo_country_by_name_cte('country_by_name') }},

ref_by_iso2 as (
    select
        country_iso2,
        country_iso3,
        country_name,
        in_africa_scope
    from {{ ref('int_ref_country') }}
    where country_iso2 is not null
),

ref_by_name as (
    select
        lower(trim(country_name)) as country_name_norm,
        country_iso2,
        country_iso3,
        country_name,
        in_africa_scope
    from {{ ref('int_ref_country') }}
    where country_name is not null
),

resolved as (
    select
        c.country_name,
        coalesce(
            upper(trim(c.country_code)),
            rc_iso.country_iso2,
            rc_name.country_iso2,
            g_name.country_iso2
        ) as country_iso2,
        coalesce(
            rc_iso.country_iso3,
            rc_name.country_iso3,
            g_iso.country_iso3,
            g_name.country_iso3,
            case when length(trim(c.country_code)) = 3 then upper(trim(c.country_code)) end
        ) as country_iso3,
        coalesce(rc_iso.in_africa_scope, rc_name.in_africa_scope, false) as in_africa_scope,
        c.model,
        c.scenario,
        c.category,
        c.subcategory,
        c.indicator,
        c.unit,
        c.year,
        c.value,
        coalesce(g_iso.geo_key, g_name.geo_key) as geo_key,
        c.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('stg_climatewatch_health') }} c
    left join ref_by_iso2 rc_iso
        on rc_iso.country_iso2 = upper(trim(c.country_code))
    left join ref_by_name rc_name
        on rc_name.country_name_norm = lower(trim(c.country_name))
    left join iso2_geo g_iso
        on g_iso.country_iso2 = coalesce(
            upper(trim(c.country_code)),
            rc_iso.country_iso2,
            rc_name.country_iso2
        )
    left join country_by_name g_name
        on g_name.country_name_norm = lower(trim(c.country_name))
)

select *
from resolved
where in_africa_scope
