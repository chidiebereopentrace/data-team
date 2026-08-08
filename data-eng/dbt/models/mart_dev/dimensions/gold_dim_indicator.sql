{{ config(materialized='table') }}

-- Gold: indicator dimension extended from staging_dev dim_indicator.

select
    indicator_key,
    indicator_key                                  as indicator_code,
    indicator_name,
    cast(null as string)                           as indicator_category,
    cast(null as string)                           as description,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from {{ ref('dim_indicator') }}
