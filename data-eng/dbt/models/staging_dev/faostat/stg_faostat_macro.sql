{{ config(materialized='table') }}

with base as (

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Macro-Economic_Indicators_Macro_Indicators' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Macro-Economic_Indicators_Macro_Indicators') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Macro-Economic_Indicators_Capital_Stock' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Macro-Economic_Indicators_Capital_Stock') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['area_code', 'item_code', 'element_code', 'year', 'source_natural_key']) }} as faostat_macro_sk,
    base.*
from base
