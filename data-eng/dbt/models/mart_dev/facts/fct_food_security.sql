{{ config(materialized='incremental', unique_key='food_security_key') }}

-- Gold: food security fact from FEWS population estimates (IPC phases, values, projections).

select
    to_hex(md5(concat(
        coalesce(fnid, country_code, ''), '|',
        coalesce(scenario_name, ''), '|',
        coalesce(phase_name, ''), '|',
        coalesce(projection_start, '')
    )))                                            as food_security_key,

    to_hex(md5(concat(
        coalesce(country_code, ''), '|',
        coalesce(country, ''), '|',
        coalesce(admin_1, '')
    )))                                            as geo_key,

    cast(
        left(coalesce(projection_start, '2000'), 4) as int64
    )                                              as time_key,

    to_hex(md5(concat(
        coalesce(indicator_name, ''), '|',
        coalesce(indicator_abbreviation, '')
    )))                                            as season_key,

    to_hex(md5('FEWS_NET_Food_insecure_population_estimates_time_series_data'))
                                                   as source_key,

    to_hex(md5(coalesce(scenario_name, '')))       as scenario_key,

    to_hex(md5(concat(
        coalesce(phase_name, ''), '|',
        coalesce(phase, '')
    )))                                            as classification_key,

    cast(null as string)                           as unit_key,
    cast(null as string)                           as audit_key,

    cast(value as numeric)                         as value,
    cast(low_value as numeric)                     as low_value,
    cast(high_value as numeric)                    as high_value,
    cast(pct_phase3 as numeric)                    as pct_phase3,
    cast(pct_phase4 as numeric)                    as pct_phase4,
    cast(pct_phase5 as numeric)                    as pct_phase5,
    cast(null as int64)                            as forecast_horizon,
    coalesce(fnid, country_code)                   as source_record_id

from {{ source('landing', 'FEWS_NET_Food_insecure_population_estimates_time_series_data') }}
where value is not null
  or low_value is not null
  or high_value is not null

{% if is_incremental() %}
  and projection_start > (select max(left(cast(time_key as string), 4)) from {{ this }})
{% endif %}
