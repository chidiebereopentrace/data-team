{{ config(materialized='table') }}

select
    household_id,
    sub_county,
    village_name,
    village_code,
    farmer_sex,
    education_level,
    housing_type,
    breed_type,
    animals_sold,
    cash_received,
    clinical_disease_score,
    positive_cases,
    incidence_rate,
    treatment_cost,
    average_herd_size,
    mortality_count,
    species,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_ilri_animal_health') }}
where household_id is not null
