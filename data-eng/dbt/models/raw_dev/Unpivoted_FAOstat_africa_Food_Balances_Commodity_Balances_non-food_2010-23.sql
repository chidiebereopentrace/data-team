{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_Balances_Commodity_Balances_non-food_2010-23') }}
