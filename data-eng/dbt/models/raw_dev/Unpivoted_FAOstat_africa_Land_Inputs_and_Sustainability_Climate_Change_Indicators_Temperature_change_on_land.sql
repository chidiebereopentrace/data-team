{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Land_Inputs_and_Sustainability_Climate_Change_Indicators_Temperature_change_on_land') }}
