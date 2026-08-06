{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Food_Value_Chain_Value_shares_by_industry_and_primary_factors') }}
