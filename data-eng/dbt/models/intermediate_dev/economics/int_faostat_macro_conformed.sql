{{ config(materialized='table') }}

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
    g.geo_key,
    m.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_macro') }} m
left join {{ ref('int_geography_conformed') }} g
    on g.geo_level = 'country'
   and g.native_id = cast(m.area_code as string)
where m.year is not null
