{{ config(materialized='table') }}

-- Gold: household fact — schema registered; populated once ILRI household
-- staging is wired in. Returns empty result set with full DDL column shape.

select
    cast(null as string)                           as household_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as person_key,
    cast(null as string)                           as audit_key,
    cast(null as numeric)                          as hh_size,
    cast(null as numeric)                          as hh_size_mae,
    cast(null as numeric)                          as land_cultivated,
    cast(null as numeric)                          as livestock_holdings,
    cast(null as numeric)                          as fies_score,
    cast(null as numeric)                          as total_income,
    cast(null as numeric)                          as farm_income,
    cast(null as numeric)                          as offfarm_income,
    cast(null as numeric)                          as food_availability,
    cast(null as int64)                            as years_farming,
    cast(null as string)                           as primary_occupation,
    cast(null as int64)                            as plots_number,
    cast(null as numeric)                          as crop_sales,
    cast(null as numeric)                          as value_crop_produce,
    cast(null as numeric)                          as value_crop_consumed,
    cast(null as numeric)                          as livestock_product_sales,
    cast(null as numeric)                          as value_livestock_production,
    cast(null as numeric)                          as value_livestock_consumed,
    cast(null as numeric)                          as food_self_sufficiency,
    cast(null as numeric)                          as total_energy_available,
    cast(null as int64)                            as crop_diversity,
    cast(null as int64)                            as livestock_diversity,
    cast(null as int64)                            as years_in_organisation,
    cast(null as string)                           as membership_type,
    cast(null as bool)                             as cooperative_member,
    cast(null as string)                           as herd_size_category,
    cast(null as int64)                            as insurance_enrolment_year,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy) where false
