{{ config(materialized='table') }}

-- Gold: humanitarian fact extended from staging_dev fact_humanitarian.

select
    fact_humanitarian_key                          as humanitarian_key,
    geography_key                                  as geo_key,
    period_key                                     as time_key,
    source_key,
    ipc_phase_key                                  as classification_key,
    cast(null as string)                           as audit_key,
    cast(null as bool)                             as is_allowing_for_assistance,
    humanitarian_value                             as assistance_value,
    fact_humanitarian_key                          as source_record_id

from {{ ref('fact_humanitarian') }}
