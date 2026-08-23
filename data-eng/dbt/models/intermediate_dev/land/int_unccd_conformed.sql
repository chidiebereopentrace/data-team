{{ config(materialized='table') }}

select
    geo_area_code,
    geo_area_name,
    series,
    series_description,
    safe_cast(time_period_start as int64) as year,
    safe_cast(value as float64) as value,
    value_type,
    time_detail,
    upper_bound,
    lower_bound,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_unccd_land_degradation') }}
where geo_area_code is not null
