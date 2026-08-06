{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Totals_and_Indicators_Emissions_totals') }}
