{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Cost_and_Affordability_of_a_Healthy_Diet_Cost_and_Affordability_of_a_Healthy_Diet') }}
