{{ config(materialized='table') }}

select
    country_name,
    country_code,
    observation_year,
    gdp_per_capita_ppp,
    ingested_at,
    'africa_gdp_ppp' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'africa_gross_domestic_product_purchasing_power_parity_bronze') }}
