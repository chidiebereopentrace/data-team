{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_World_Census_of_Agriculture_Structural_data_from_agricultural_censuses') }}
