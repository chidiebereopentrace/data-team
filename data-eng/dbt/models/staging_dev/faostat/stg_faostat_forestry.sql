{{ config(materialized='table') }}

with base as (

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Forestry_Forestry_Production_and_Trade' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Forestry_Forestry_Production_and_Trade') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as partner_country_code,
    cast(null as string) as partner_country_code_m49,
    cast(null as string) as partner_countries,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Forestry_Pulp_and_paper_capacities_survey' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Forestry_Pulp_and_paper_capacities_survey') }}

union all

select
    Reporter_Country_Code as area_code,
    Reporter_Country_Code_M49 as area_code_m49,
    Reporter_Countries as country_name,
    Partner_Country_Code as partner_country_code,
    Partner_Country_Code_M49 as partner_country_code_m49,
    Partner_Countries as partner_countries,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Discontinued_archives_and_data_series_Forestry_Trade_Flows' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Discontinued_archives_and_data_series_Forestry_Trade_Flows') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['area_code', 'partner_country_code', 'item_code', 'element_code', 'year', 'source_natural_key']) }} as faostat_forestry_sk,
    base.*
from base
