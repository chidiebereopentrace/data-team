{{ config(materialized='table') }}

-- Gold: animal health fact — schema registered, populated once ILRI household
-- staging is wired in. Returns empty result set with full DDL column shape.

select
    cast(null as string)                           as animal_health_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as season_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as livestock_key,
    cast(null as string)                           as disease_key,
    cast(null as string)                           as pest_key,
    cast(null as string)                           as person_key,
    cast(null as string)                           as unit_key,
    cast(null as string)                           as audit_key,
    cast(null as numeric)                          as clinical_disease_score,
    cast(null as numeric)                          as incidence_rate,
    cast(null as int64)                            as positive_cases,
    cast(null as numeric)                          as treatment_cost,
    cast(null as int64)                            as mortality_count,
    cast(null as numeric)                          as application_frequency,
    cast(null as numeric)                          as eiq_score,
    cast(null as numeric)                          as eiq_field_use,
    cast(null as bool)                             as treatment_misuse_flag,
    cast(null as string)                           as deworming_practice,
    cast(null as int64)                            as vet_checkup_frequency,
    cast(null as int64)                            as livestock_births,
    cast(null as int64)                            as livestock_purchased,
    cast(null as int64)                            as livestock_sold,
    cast(null as numeric)                          as monthly_caseload,
    cast(null as string)                           as treatment_protocol,
    cast(null as string)                           as diagnostic_test_used,
    cast(null as bool)                             as antibiotic_combination_used,
    cast(null as int64)                            as total_samples,
    cast(null as int64)                            as positive_samples,
    cast(null as numeric)                          as mean_cfu_log,
    cast(null as numeric)                          as prevalence_pct,
    cast(null as int64)                            as parity_number,
    cast(null as date)                             as calving_date,
    cast(null as bool)                             as is_lactating,
    cast(null as int64)                            as lactation_months,
    cast(null as date)                             as next_calving_date,
    cast(null as string)                           as source_record_id,
    cast(null as string)                           as qc_flag

from (select 1 as _dummy) where false
