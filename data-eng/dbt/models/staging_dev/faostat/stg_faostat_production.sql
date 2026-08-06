{{ config(materialized='view') }}

select
    area_code,
    area_code_m49,
    area as country_name,
    item_code,
    item_code_cpc,
    item as product_name,
    element_code,
    element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_production_Crops_and_livestock' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_production_Crops_and_livestock') }}

union all

select
    area_code,
    area_code_m49,
    area as country_name,
    item_code,
    item_code_cpc,
    item as product_name,
    element_code,
    element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_production_Production Indices' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_production_Production Indices') }}

union all

select
    area_code,
    area_code_m49,
    area as country_name,
    item_code,
    item_code_cpc,
    item as product_name,
    element_code,
    element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_production_Value_of_Agricultural_Production' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_production_Value_of_Agricultural_Production') }}
