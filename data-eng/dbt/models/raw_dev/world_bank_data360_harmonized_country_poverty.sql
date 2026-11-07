{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'world_bank_data360_harmonized_country_poverty') }}
