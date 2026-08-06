{{ config(materialized='table') }}

-- FAOSTAT Agrifood Systems Emissions (farm-gate, land-use, pre/post, totals/indicators).
-- Branches cast missing columns to null so heterogeneous raw schemas union cleanly.

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    Source_Code as source_code,
    Source as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Farm_gate_Emissions_from_Crops' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Farm_gate_Emissions_from_Crops') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    Source_Code as source_code,
    Source as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Farm_gate_Emissions_from_Livestock' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Farm_gate_Emissions_from_Livestock') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as source_code,
    cast(null as string) as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Farm_gate_Emissions_from_Energy_use_in_agriculture' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Farm_gate_Emissions_from_Energy_use_in_agriculture') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    Source_Code as source_code,
    Source as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Land_use_and_change_Emissions_from_Fires' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Land_use_and_change_Emissions_from_Fires') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    Source_Code as source_code,
    Source as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Land_use_and_change_Emissions_from_Forests' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Land_use_and_change_Emissions_from_Forests') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    Source_Code as source_code,
    Source as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Land_use_and_change_Emissions_from_Drained_organic_soils' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Land_use_and_change_Emissions_from_Drained_organic_soils') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as source_code,
    cast(null as string) as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Pre_and_post_agricultural_production_Emissions_from_pre_and_post_agricultural_production' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Pre_and_post_agricultural_production_Emissions_from_pre_and_post_agricultural_production') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    Source_Code as source_code,
    Source as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Totals_and_Indicators_Emissions_totals' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Totals_and_Indicators_Emissions_totals') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as source_code,
    cast(null as string) as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Totals_and_Indicators_Emissions_intensities' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Totals_and_Indicators_Emissions_intensities') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as source_code,
    cast(null as string) as source,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Totals_and_Indicators_Emissions_indicators' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Totals_and_Indicators_Emissions_indicators') }}
