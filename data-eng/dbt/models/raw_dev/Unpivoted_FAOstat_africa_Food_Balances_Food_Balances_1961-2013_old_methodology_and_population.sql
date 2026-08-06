{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_Balances_Food_Balances_1961-2013_old_methodology_and_population') }}
