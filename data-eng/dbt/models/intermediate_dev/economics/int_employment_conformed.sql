{{ config(materialized='table') }}

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
    g.geo_key,
    e.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_population_employment') }} e
left join {{ ref('int_geography_conformed') }} g
    on g.geo_level = 'country'
   and g.native_id = cast(e.area_code as string)
where e.year is not null
