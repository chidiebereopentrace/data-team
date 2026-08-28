{{ config(materialized='table') }}

with country_by_native_id as (
    select
        native_id,
        geo_key,
        country_iso3
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
    e.area_code,
    e.area_code_m49,
    e.country_name,
    e.item_code,
    e.item,
    e.element_code,
    e.element,
    e.indicator_code,
    e.indicator,
    e.source_code,
    e.source,
    e.sex_code,
    e.sex,
    e.year,
    e.unit,
    safe_cast(e.value as float64) as value,
    coalesce(g_nat.geo_key, g_iso.geo_key) as geo_key,
    coalesce(
        case when g_nat.geo_key is not null then g_nat.country_iso3 end,
        case when g_iso.geo_key is not null then g_iso.country_iso3 end,
        iso.country_iso3
    ) as country_iso3,
    e.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_population_employment') }} e
left join {{ ref('int_faostat_area_iso') }} iso
    on iso.area_code = cast(e.area_code as string)
left join country_by_native_id g_nat
    on g_nat.native_id = cast(e.area_code as string)
left join country_by_iso3 g_iso
    on g_iso.country_iso3 = upper(trim(iso.country_iso3))
where e.year is not null
