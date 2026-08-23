{{ config(materialized='table') }}

select
    fnid,
    country,
    country_code,
    admin_0,
    admin_1,
    admin_2,
    admin_3,
    admin_4,
    geographic_unit_name,
    fewsnet_region,
    measure_type,
    phase_code,
    phase_name,
    classification_scale,
    scenario_name,
    value,
    low_value,
    high_value,
    pct_phase3,
    pct_phase4,
    pct_phase5,
    is_allowing_for_assistance,
    year,
    month,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_fews_food_security') }}
where measure_type is not null
