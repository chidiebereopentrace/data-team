{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Agricultural_Science_and_Technology_ASTI-Researchers') }}
