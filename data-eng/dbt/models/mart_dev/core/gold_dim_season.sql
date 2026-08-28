{{ config(materialized='table') }}

-- Gold: full season dimension extended from staging_dev dim_season.

select
    season_key,
    season_key                                     as season_natural_key,
    season_name,
    season_name                                    as season_name_std,
    cast(null as string)                           as season_type,
    cast(null as int64)                            as typical_start_month,
    cast(null as int64)                            as typical_end_month,
    cast(null as string)                           as applies_to_region,
    cast(null as string)                           as source_system,
    cast(null as string)                           as description,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from {{ ref('dim_season') }}
