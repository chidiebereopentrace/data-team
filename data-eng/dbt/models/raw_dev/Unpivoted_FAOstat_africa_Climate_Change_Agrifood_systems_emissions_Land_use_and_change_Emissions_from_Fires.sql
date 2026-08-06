{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Land_use_and_change_Emissions_from_Fires') }}
