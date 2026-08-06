{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'enriched_isda_soil_property_Africa') }}
