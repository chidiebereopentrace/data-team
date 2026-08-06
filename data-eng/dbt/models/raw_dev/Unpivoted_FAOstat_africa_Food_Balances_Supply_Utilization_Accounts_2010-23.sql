{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_Balances_Supply_Utilization_Accounts_2010-23') }}
