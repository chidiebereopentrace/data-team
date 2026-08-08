{{ config(materialized='table') }}

-- Gold: farm practice dimension extended from staging_dev dim_farm_practice.

select
    practice_key                                   as farm_practice_key,
    practice_key                                   as practice_code,
    practice_name,
    cast(null as string)                           as practice_category,
    cast(null as string)                           as description

from {{ ref('dim_farm_practice') }}
