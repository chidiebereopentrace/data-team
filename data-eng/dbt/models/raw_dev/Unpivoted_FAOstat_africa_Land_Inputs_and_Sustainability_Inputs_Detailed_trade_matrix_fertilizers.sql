{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Inputs_Detailed_trade_matrix_fertilizers') }}
