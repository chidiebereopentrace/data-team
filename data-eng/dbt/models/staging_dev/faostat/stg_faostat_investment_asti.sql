{{ config(materialized='table') }}

-- FAOSTAT Investment + ASTI.
-- Investment rows: item/element; ASTI rows: indicator/institution (+ degree/sex or cost_category).
-- Development Flows: Recipient → area_*; Donor/Purpose retained for grain.

with base as (

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as donor_code,
    cast(null as string) as donor_code_m49,
    cast(null as string) as donor,
    cast(null as string) as purpose_code,
    cast(null as string) as purpose,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as indicator_code,
    cast(null as string) as indicator,
    cast(null as string) as institution_code,
    cast(null as string) as institution,
    cast(null as string) as degree_code,
    cast(null as string) as degree,
    cast(null as string) as sex_code,
    cast(null as string) as sex,
    cast(null as string) as cost_category_code,
    cast(null as string) as cost_category,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Investment_Country_Investment_Statistics_Profile' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Investment_Country_Investment_Statistics_Profile') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as donor_code,
    cast(null as string) as donor_code_m49,
    cast(null as string) as donor,
    cast(null as string) as purpose_code,
    cast(null as string) as purpose,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as indicator_code,
    cast(null as string) as indicator,
    cast(null as string) as institution_code,
    cast(null as string) as institution,
    cast(null as string) as degree_code,
    cast(null as string) as degree,
    cast(null as string) as sex_code,
    cast(null as string) as sex,
    cast(null as string) as cost_category_code,
    cast(null as string) as cost_category,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Investment_Credit_to_Agriculture' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Investment_Credit_to_Agriculture') }}

union all

select
    Recipient_Country_Code as area_code,
    Recipient_Country_Code_M49 as area_code_m49,
    Recipient_Country as country_name,
    Donor_Code as donor_code,
    Donor_Code_M49 as donor_code_m49,
    Donor as donor,
    Purpose_Code as purpose_code,
    Purpose as purpose,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as indicator_code,
    cast(null as string) as indicator,
    cast(null as string) as institution_code,
    cast(null as string) as institution,
    cast(null as string) as degree_code,
    cast(null as string) as degree,
    cast(null as string) as sex_code,
    cast(null as string) as sex,
    cast(null as string) as cost_category_code,
    cast(null as string) as cost_category,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Investment_Development_Flows_to_Agriculture' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Investment_Development_Flows_to_Agriculture') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as donor_code,
    cast(null as string) as donor_code_m49,
    cast(null as string) as donor,
    cast(null as string) as purpose_code,
    cast(null as string) as purpose,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as indicator_code,
    cast(null as string) as indicator,
    cast(null as string) as institution_code,
    cast(null as string) as institution,
    cast(null as string) as degree_code,
    cast(null as string) as degree,
    cast(null as string) as sex_code,
    cast(null as string) as sex,
    cast(null as string) as cost_category_code,
    cast(null as string) as cost_category,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Investment_Foreign_Direct_Investment_FDI' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Investment_Foreign_Direct_Investment_FDI') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as donor_code,
    cast(null as string) as donor_code_m49,
    cast(null as string) as donor,
    cast(null as string) as purpose_code,
    cast(null as string) as purpose,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as indicator_code,
    cast(null as string) as indicator,
    cast(null as string) as institution_code,
    cast(null as string) as institution,
    cast(null as string) as degree_code,
    cast(null as string) as degree,
    cast(null as string) as sex_code,
    cast(null as string) as sex,
    cast(null as string) as cost_category_code,
    cast(null as string) as cost_category,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Investment_Government_Expenditure' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Investment_Government_Expenditure') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as donor_code,
    cast(null as string) as donor_code_m49,
    cast(null as string) as donor,
    cast(null as string) as purpose_code,
    cast(null as string) as purpose,
    cast(null as string) as item_code,
    cast(null as string) as item,
    cast(null as string) as element_code,
    cast(null as string) as element,
    Indicator_Code as indicator_code,
    Indicator as indicator,
    Institution_Code as institution_code,
    Institution as institution,
    Degree_Code as degree_code,
    Degree as degree,
    Sex_Code as sex_code,
    Sex as sex,
    cast(null as string) as cost_category_code,
    cast(null as string) as cost_category,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Agricultural_Science_and_Technology_ASTI-Researchers' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Agricultural_Science_and_Technology_ASTI-Researchers') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as donor_code,
    cast(null as string) as donor_code_m49,
    cast(null as string) as donor,
    cast(null as string) as purpose_code,
    cast(null as string) as purpose,
    cast(null as string) as item_code,
    cast(null as string) as item,
    cast(null as string) as element_code,
    cast(null as string) as element,
    Indicator_Code as indicator_code,
    Indicator as indicator,
    Institution_Code as institution_code,
    Institution as institution,
    cast(null as string) as degree_code,
    cast(null as string) as degree,
    cast(null as string) as sex_code,
    cast(null as string) as sex,
    Cost_Category_Code as cost_category_code,
    Cost_Category as cost_category,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Agricultural_Science_and_Technology_ASTI-Expenditures' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Agricultural_Science_and_Technology_ASTI-Expenditures') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['area_code', 'donor_code', 'purpose_code', 'item_code', 'element_code', 'indicator_code', 'institution', 'degree_code', 'sex_code', 'cost_category_code', 'year', 'source_natural_key']) }} as faostat_investment_asti_sk,
    base.*
from base
