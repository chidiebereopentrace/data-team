{{ config(materialized='table') }}

-- FAOSTAT Prices: producer, CPI, deflators, exchange rates.
-- Shared contract includes item + months + currency columns; null where a source lacks them.

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    Item_Code_CPC as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    Months_Code as months_code,
    Months as months,
    cast(null as string) as iso_currency_code,
    cast(null as string) as currency,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Prices_Producer_Prices' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Prices_Producer_Prices') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    Months_Code as months_code,
    Months as months,
    cast(null as string) as iso_currency_code,
    cast(null as string) as currency,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Prices_Consumer_Price_Indices' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Prices_Consumer_Price_Indices') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    Item_Code as item_code,
    cast(null as string) as item_code_cpc,
    Item as product_name,
    Element_Code as element_code,
    Element as element,
    cast(null as string) as months_code,
    cast(null as string) as months,
    cast(null as string) as iso_currency_code,
    cast(null as string) as currency,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Prices_Deflators' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Prices_Deflators') }}

union all

select
    Area_Code as area_code,
    Area_Code_M49 as area_code_m49,
    Area as country_name,
    cast(null as string) as item_code,
    cast(null as string) as item_code_cpc,
    cast(null as string) as product_name,
    Element_Code as element_code,
    Element as element,
    Months_Code as months_code,
    Months as months,
    ISO_Currency_Code as iso_currency_code,
    Currency as currency,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Prices_Exchange_rates' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Prices_Exchange_rates') }}
