{{ config(materialized='table') }}

with iso2_geo as (
    select
        upper(trim(country_iso2)) as country_iso2,
        geo_key
    from {{ ref('int_geography_conformed') }}
    where country_iso2 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso2))
        order by case when capital_status is not null then 0 else 1 end, geo_key
    ) = 1
)

select
    n.country_code,
    n.country_name,
    n.admin_region,
    n.latitude,
    n.longitude,
    n.elevation_meters,
    n.par_solar_at_noon,
    n.shortwave_irradiance_at_noon,
    n.uva_radiation_at_noon,
    n.uvb_radiation_at_noon,
    n.observation_time,
    coalesce(g_city.geo_key, g_country.geo_key) as geo_key,
    n.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_nasa_power') }} n
left join {{ ref('int_geography_conformed') }} g_city
    on n.latitude is not null
   and n.longitude is not null
   and g_city.geo_level = 'city'
   and round(n.latitude, 2) = round(g_city.latitude, 2)
   and round(n.longitude, 2) = round(g_city.longitude, 2)
left join iso2_geo g_country
    on upper(trim(n.country_code)) = g_country.country_iso2
where (
        n.latitude is null
        or n.longitude is null
        or (
            n.longitude between -25 and 60
            and n.latitude between -35 and 38
        )
    )
