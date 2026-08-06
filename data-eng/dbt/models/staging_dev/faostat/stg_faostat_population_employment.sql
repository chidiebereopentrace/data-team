{{ config(materialized='table') }}

-- FAOSTAT Population & Employment.
-- Population: item/element; Employment: indicator/source/sex + element.

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as indicator_code,
    cast(null as string) as indicator,
    cast(null as string) as source_code,
    cast(null as string) as source,
    cast(null as string) as sex_code,
    cast(null as string) as sex,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Population_and_Employment_Annual_population' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Population_and_Employment_Annual_population') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as item_code,
    cast(null as string) as item,
    Element_Code as element_code,
    Element as element,
    Indicator_Code as indicator_code,
    Indicator as indicator,
    Source_Code as source_code,
    Source as source,
    Sex_Code as sex_code,
    Sex as sex,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Population_and_Employment_Employment_Indicators_Employment_Indicators_Agriculture_and_agrifood_systems' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Population_and_Employment_Employment_Indicators_Employment_Indicators_Agriculture_and_agrifood_systems') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as item_code,
    cast(null as string) as item,
    Element_Code as element_code,
    Element as element,
    Indicator_Code as indicator_code,
    Indicator as indicator,
    Source_Code as source_code,
    Source as source,
    Sex_Code as sex_code,
    Sex as sex,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Population_and_Employment_Employment_Indicators_Employment_Indicators_Rural' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Population_and_Employment_Employment_Indicators_Employment_Indicators_Rural') }}
