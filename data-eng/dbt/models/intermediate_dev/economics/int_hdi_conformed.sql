{{ config(materialized='table') }}

with iso3_geo as (
    select
        upper(trim(country_iso3)) as country_iso3,
        geo_key
    from {{ ref('int_geography_conformed') }}
    where country_iso3 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso3))
        order by case when capital_status is not null then 0 else 1 end, geo_key
    ) = 1
)

select
    h.country as country_name,
    upper(trim(h.country_code)) as country_iso3,
    safe_cast(h.year as int64) as year,
    safe_cast(h.hdi_value as float64) as hdi_value,
    g.geo_key,
    h.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_africa_hdi') }} h
left join iso3_geo g
    on g.country_iso3 = upper(trim(h.country_code))
where h.hdi_value is not null
  and safe_cast(h.hdi_value as float64) is not null
  and safe_cast(h.year as int64) is not null
