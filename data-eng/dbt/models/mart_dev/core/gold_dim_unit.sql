{{ config(materialized='table') }}

-- Gold: full unit dimension extended from staging_dev dim_unit.

select
    unit_key,
    unit_name                                      as unit_code,
    unit_name,
    cast(null as string)                           as unit_type,
    cast(null as string)                           as common_unit,
    cast(null as string)                           as currency_code,
    cast(null as string)                           as description

from {{ ref('dim_unit') }}
