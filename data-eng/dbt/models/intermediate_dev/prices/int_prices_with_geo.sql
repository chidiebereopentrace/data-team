{{ config(materialized='table') }}

with {{ geo_country_by_name_cte('country_by_name') }},
{{ geo_country_by_iso3_cte('country_by_iso3') }},

prices as (
    select * from {{ ref('int_prices_harmonised') }}
),

wfp_with_iso3 as (
    select
        p.*,
        coalesce(
            rc_name.country_iso3,
            rc_alias.country_iso3,
            {{ geo_faostat_country_name_iso3('p.country') }}
        ) as resolved_country_iso3
    from prices p
    left join {{ ref('int_ref_country') }} rc_name
        on lower(trim(rc_name.country_name)) = lower(trim(p.country))
    left join {{ ref('int_ref_country') }} rc_alias
        on rc_alias.country_iso3 = {{ geo_faostat_country_name_iso3('p.country') }}
    where p.price_source = 'wfp'
),

faostat_geo as (
    select
        p.*,
        g.geo_key,
        g.country_iso2,
        g.country_iso3,
        g.country_name as geo_country_name
    from prices p
    left join {{ ref('int_geography_conformed') }} g
        on p.price_source = 'faostat'
       and g.geo_level = 'country'
       and g.native_id = cast(p.area_code as string)
),

fews_geo as (
    select
        p.*,
        coalesce(g_admin2.geo_key, g_admin1.geo_key, g_country.geo_key) as geo_key,
        coalesce(g_admin2.country_iso2, g_admin1.country_iso2, g_country.country_iso2) as country_iso2,
        coalesce(g_admin2.country_iso3, g_admin1.country_iso3, g_country.country_iso3) as country_iso3,
        coalesce(g_admin2.country_name, g_admin1.country_name, g_country.country_name) as geo_country_name
    from prices p
    left join {{ ref('int_geography_conformed') }} g_admin2
        on p.price_source = 'fews'
       and {{ geo_fews_admin2_join('p', 'g_admin2') }}
    left join {{ ref('int_geography_conformed') }} g_admin1
        on p.price_source = 'fews'
       and {{ geo_fews_admin1_join('p', 'g_admin1') }}
    left join {{ ref('int_geography_conformed') }} g_country
        on p.price_source = 'fews'
       and {{ geo_fews_country_join('p', 'g_country') }}
),

wfp_geo as (
    select
        p.* except (resolved_country_iso3),
        coalesce(g_admin1.geo_key, g_iso.geo_key, g_name.geo_key) as geo_key,
        coalesce(g_admin1.country_iso2, g_iso.country_iso2, g_name.country_iso2) as country_iso2,
        coalesce(g_admin1.country_iso3, g_iso.country_iso3, g_name.country_iso3, p.resolved_country_iso3) as country_iso3,
        coalesce(g_admin1.country_name, g_iso.country_name, g_name.country_name) as geo_country_name
    from wfp_with_iso3 p
    left join {{ ref('int_geography_conformed') }} g_admin1
        on {{ geo_wfp_admin1_join_by_iso3('p', 'g_admin1', 'p.resolved_country_iso3') }}
    left join country_by_iso3 g_iso
        on upper(trim(p.resolved_country_iso3)) = g_iso.country_iso3_norm
    left join country_by_name g_name
        on lower(trim(p.country)) = g_name.country_name_norm
)

select * from faostat_geo where price_source = 'faostat'
union all
select * from fews_geo where price_source = 'fews'
union all
select * from wfp_geo where price_source = 'wfp'
