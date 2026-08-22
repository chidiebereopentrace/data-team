{{ config(materialized='table') }}

with base as (

select
    Recipient_Country_Code as area_code,
    Recipient_Country_Code_M49 as area_code_m49,
    Recipient_Country as country_name,
    Item_Code as item_code,
    Item as item,
    Element_Code as element_code,
    Element as element,
    year,
    Unit as unit,
    value,
    'Unpivoted_FAOstat_africa_Discontinued_archives_and_data_series_Food_Aid_Shipments_WFP' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Discontinued_archives_and_data_series_Food_Aid_Shipments_WFP') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['area_code', 'item_code', 'element_code', 'year', 'source_natural_key']) }} as faostat_food_aid_sk,
    base.*
from base
