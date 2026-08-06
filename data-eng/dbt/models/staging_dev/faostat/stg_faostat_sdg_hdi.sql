{{ config(materialized='table') }}

-- FAOSTAT SDG + related socio indicators.
-- FVC uses Food_Value/Industry/Factor (not Item); CoAHD has Release; Census has Census_Year.
-- Africa HDI lives in socio_economic/stg_africa_hdi.

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_SDG as item_code_sdg,
    Item as item,
    cast(null as string) as food_value_code,
    cast(null as string) as food_value,
    cast(null as string) as industry_code,
    cast(null as string) as industry,
    cast(null as string) as factor_code,
    cast(null as string) as factor,
    cast(null as string) as release_code,
    cast(null as string) as release,
    cast(null as string) as census_year_code,
    cast(null as string) as census_year,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_SDG_Indicators_SDG_Indicators' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_SDG_Indicators_SDG_Indicators') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_sdg,
    Item as item,
    cast(null as string) as food_value_code,
    cast(null as string) as food_value,
    cast(null as string) as industry_code,
    cast(null as string) as industry,
    cast(null as string) as factor_code,
    cast(null as string) as factor,
    Release_Code as release_code,
    Release as release,
    cast(null as string) as census_year_code,
    cast(null as string) as census_year,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Cost_and_Affordability_of_a_Healthy_Diet_Cost_and_Affordability_of_a_Healthy_Diet' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Cost_and_Affordability_of_a_Healthy_Diet_Cost_and_Affordability_of_a_Healthy_Diet') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as item_code,
    cast(null as string) as item_code_sdg,
    cast(null as string) as item,
    Food_Value_Code as food_value_code,
    Food_Value as food_value,
    Industry_Code as industry_code,
    Industry as industry,
    Factor_Code as factor_code,
    Factor as factor,
    cast(null as string) as release_code,
    cast(null as string) as release,
    cast(null as string) as census_year_code,
    cast(null as string) as census_year,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Food_Value_Chain_Value_shares_by_industry_and_primary_factors' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_Value_Chain_Value_shares_by_industry_and_primary_factors') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_sdg,
    Item as item,
    cast(null as string) as food_value_code,
    cast(null as string) as food_value,
    cast(null as string) as industry_code,
    cast(null as string) as industry,
    cast(null as string) as factor_code,
    cast(null as string) as factor,
    cast(null as string) as release_code,
    cast(null as string) as release,
    Census_Year_Code as census_year_code,
    Census_Year as census_year,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_World_Census_of_Agriculture_Structural_data_from_agricultural_censuses' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_World_Census_of_Agriculture_Structural_data_from_agricultural_censuses') }}
