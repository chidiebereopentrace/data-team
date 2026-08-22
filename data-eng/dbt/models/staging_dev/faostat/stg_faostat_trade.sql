{{ config(materialized='table') }}

-- FAOSTAT Trade: crops & livestock products, trade indices, and indicators.
-- Indicators source uses Indicator_Code/Indicator; mapped to element_code/element.

with base as (

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Trade_Crops_and_livestock_products' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Trade_Crops_and_livestock_products') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Trade_Trade_Indices' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Trade_Trade_Indices') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as product_name,
    Indicator_Code as element_code,
    Indicator as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Trade_Crops_and_livestock_products_indicators' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Trade_Crops_and_livestock_products_indicators') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['area_code', 'item_code', 'element_code', 'year', 'source_natural_key']) }} as faostat_trade_sk,
    base.*
from base
