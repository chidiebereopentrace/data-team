{{ config(materialized='table') }}

with country_by_native_id as (
    select
        native_id,
        geo_key
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'country'
      and native_id is not null
    qualify row_number() over (
        partition by native_id
        order by population desc nulls last, geo_key
    ) = 1
),

country_by_iso3 as (
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
    m.area_code,
    m.area_code_m49,
    m.country_name,
    m.item_code,
    m.item,
    m.element_code,
    m.element,
    m.year,
    m.unit,
    safe_cast(m.value as float64) as value,
    case
        when lower(m.element) like '%share of gdp%' or m.unit = '%' then 'share_of_gdp'
        when lower(m.element) like '%growth%' then 'annual_growth'
        when lower(m.element) like '%2015%' or lower(m.unit) like '%2015%' then 'constant_price'
        else 'current_price'
    end as measurement_form,
    coalesce(g_nat.geo_key, g_iso.geo_key) as geo_key,
    m.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_macro') }} m
left join {{ ref('int_faostat_area_iso') }} iso
    on iso.area_code = cast(m.area_code as string)
left join country_by_native_id g_nat
    on g_nat.native_id = cast(m.area_code as string)
left join country_by_iso3 g_iso
    on g_iso.country_iso3 = upper(trim(iso.country_iso3))
where m.year is not null
