{{ config(materialized='table') }}

-- FAOSTAT Land, Inputs & Sustainability.
-- Trade matrix: Reporter → area_*; Partner retained for grain.
-- Temperature: months_* instead of item_*.

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Fertilizers_by_Nutrient' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Fertilizers_by_Nutrient') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Fertilizers_by_Product' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Fertilizers_by_Product') }}

union all

select
    Reporter_Country_Code as area_code,
    Reporter_Country_Code_M49 as area_code_m49,
    Reporter_Countries as country_name,
    Partner_Country_Code as partner_country_code,
    Partner_Country_Code_M49 as partner_country_code_m49,
    Partner_Countries as partner_countries,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Detailed_trade_matrix_fertilizers' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Detailed_trade_matrix_fertilizers') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Pesticides_Use' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Pesticides_Use') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Pesticides_Trade' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Pesticides_Trade') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Livestock_Manure' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Livestock_Manure') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Land_Land_Cover' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Land_Land_Cover') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Land_Land_Use' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Land_Land_Use') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    cast(null as string) as item_code,
    cast(null as string) as item_code_cpc,
    cast(null as string) as item,
    Months_Code as months_code,
    Months as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Climate_Change_Indicators_Temperature_change_on_land' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Climate_Change_Indicators_Temperature_change_on_land') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Sustainability_Indicators_Bioenergy' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Sustainability_Indicators_Bioenergy') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Sustainability_Indicators_Cropland_Nutrient_Balance' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Sustainability_Indicators_Cropland_Nutrient_Balance') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as item,
    cast(null as string) as months_code,
    cast(null as string) as months,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Sustainability_Indicators_Livestock_Patterns' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Sustainability_Indicators_Livestock_Patterns') }}
