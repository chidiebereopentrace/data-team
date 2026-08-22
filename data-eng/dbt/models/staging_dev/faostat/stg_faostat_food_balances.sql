{{ config(materialized='table') }}

-- FAOSTAT Food Balances: FBS series use Item_Code_FBS; SUA / commodity balances use Item_Code_CPC.
-- Shared contract includes both; null-cast where a source lacks the column.

with base as (

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_FBS as item_code_fbs,
    cast(null as string) as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Food Balances_Food_Balances_2010-23' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food Balances_Food_Balances_2010-23') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_fbs,
    Item_Code_CPC as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Food_Balances_Supply_Utilization_Accounts_2010-23' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_Balances_Supply_Utilization_Accounts_2010-23') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_fbs,
    Item_Code_CPC as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Food_Balances_Commodity_Balances_non-food_2010-23' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_Balances_Commodity_Balances_non-food_2010-23') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_FBS as item_code_fbs,
    cast(null as string) as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Food_Balances_Food_Balances_1961-2013_old_methodology_and_population' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_Balances_Food_Balances_1961-2013_old_methodology_and_population') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['area_code', 'item_code', 'element_code', 'year', 'source_natural_key']) }} as faostat_food_balances_sk,
    base.*
from base
