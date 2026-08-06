{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_and_Diet_Availability_based_on_supply_utilization_accounts') }}
