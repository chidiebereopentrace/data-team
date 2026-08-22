{{ config(materialized='table') }}

select
    {{ dbt_utils.generate_surrogate_key(['country_code', 'observation_year']) }} as africa_gdp_ppp_sk,
    country_name,
    country_code,
    observation_year,
    gdp_per_capita_ppp,
    ingested_at,
    'africa_gdp_ppp' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'africa_gross_domestic_product_purchasing_power_parity_bronze') }}
