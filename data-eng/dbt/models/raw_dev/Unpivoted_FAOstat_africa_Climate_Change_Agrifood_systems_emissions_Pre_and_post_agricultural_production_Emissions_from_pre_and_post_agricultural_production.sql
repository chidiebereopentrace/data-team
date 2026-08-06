{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Climate_Change_Agrifood_systems_emissions_Pre_and_post_agricultural_production_Emissions_from_pre_and_post_agricultural_production') }}
