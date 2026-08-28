{{ config(materialized='table') }}

{# Long-form FAOSTAT production — one row per area/item/element/year/source. #}

select
    area_code,
    area_code_m49,
    country_name,
    item_code,
    item_code_cpc,
    product_name,
    element_code,
    element,
    year,
    unit,
    value,
    source_natural_key,
    case source_natural_key
        when 'Unpivoted_FAOstat_africa_production_Crops_and_livestock' then 'physical'
        when 'Unpivoted_FAOstat_africa_production_Production Indices' then 'index'
        when 'Unpivoted_FAOstat_africa_production_Value_of_Agricultural_Production' then 'gross_value'
    end as production_grain,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_production') }}
where year is not null
  and value is not null
