{{ config(materialized='table') }}

-- Gold: technology dimension extended from staging_dev dim_technology.

select
    technology_key,
    technology_key                                 as technology_natural_key,
    technology_name,
    cast(null as string)                           as technology_category,
    cast(null as string)                           as subcategory,
    cast(null as string)                           as description,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from {{ ref('dim_technology') }}
