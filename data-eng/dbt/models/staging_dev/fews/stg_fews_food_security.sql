{{ config(materialized='table') }}

with population as (
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
        cast(phase as string) as phase_code,
        phase_name,
        cast(null as string) as classification_scale,
        scenario_name,
        value,
        low_value,
        high_value,
        pct_phase3,
        pct_phase4,
        pct_phase5,
        cast(null as boolean) as is_allowing_for_assistance,
        extract(year from cast(projection_start as date)) as year,
        extract(month from cast(projection_start as date)) as month,
        'population' as measure_type,
        'FEWS_NET_Food_insecure_population_estimates' as source_natural_key
    from {{ source('raw_dev', 'FEWS_NET_Food_insecure_population_estimates_time_series_data') }}
),

classifications as (
    select
        fnid,
        country,
        country_code,
        cast(null as string) as admin_0,
        cast(null as string) as admin_1,
        cast(null as string) as admin_2,
        cast(null as string) as admin_3,
        cast(null as string) as admin_4,
        geographic_unit_name,
        fewsnet_region,
        cast(null as string) as phase_code,
        cast(null as string) as phase_name,
        classification_scale,
        scenario_name,
        value,
        cast(null as float64) as low_value,
        cast(null as float64) as high_value,
        cast(pct_phase3 as float64) as pct_phase3,
        cast(pct_phase4 as float64) as pct_phase4,
        cast(pct_phase5 as float64) as pct_phase5,
        is_allowing_for_assistance,
        extract(year from cast(projection_start as date)) as year,
        extract(month from cast(projection_start as date)) as month,
        'classification' as measure_type,
        'FEWS_NET_food_security_classifications' as source_natural_key
    from {{ source('raw_dev', 'FEWS_NET_food_security_classifications_time_series_data') }}
),

base as (
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
        measure_type,
        source_natural_key,
        current_timestamp() as loaded_at
    from population

    union all

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
        measure_type,
        source_natural_key,
        current_timestamp() as loaded_at
    from classifications
)

select
    {{ dbt_utils.generate_surrogate_key(['fnid', 'year', 'month', 'measure_type', 'source_natural_key']) }} as fews_food_security_sk,
    base.*
from base
