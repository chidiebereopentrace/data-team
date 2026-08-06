{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Forestry_Forestry_Production_and_Trade') }}
