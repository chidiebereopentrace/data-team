{{ config(materialized='table') }}

-- ILRI Animal Health: Lira pig clinical, Acaricide cattle, AMUSE Ethiopia.

with base as (
    select
        household_code as household_id,
        sub_county,
        village_name,
        village_code,
        farmer_sex_1_male_0_female as farmer_sex,
        farmer_education_level_0_never_attended_1_primary_2_secondary_3_tertiary_4_post_graduate as education_level,
        pig_house_type_1_housed_0_tethered as housing_type,
        pig_breed_1_improved_0_local as breed_type,
        safe_cast(total_no_of_pigs_sold_during_study as float64) as animals_sold,
        safe_cast(total_cash_received_from_pigs_sold_during_study_x1000_ugx as float64) as cash_received,
        safe_cast(median_clinical_disease_scores_cds_per_farm as float64) as clinical_disease_score,
        safe_cast(total_positive_cases_all_pathogens as float64) as positive_cases,
        safe_cast(incidence_rate_ir_id_total_no_of_new_cases_infections_per_no_of_pigs_at_risk_in_given_time_period_id_col_n_col_m_x_col_o_units_pig_month_ as float64) as incidence_rate,
        safe_cast(total_treatment_costs_during_study_value_x1000_ugx as float64) as treatment_cost,
        safe_cast(average_herd_size_total_no_of_pigs_for_all_visits_no_of_samplings_visits_ as float64) as average_herd_size,
        safe_cast(total_no_of_pigs_that_died_in_each_farm_during_the_study as float64) as mortality_count,
        'pig' as species,
        'ilri_impact_assessment_data_lira_uganda_nov25' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'ilri_impact_assessment_data_lira_uganda_nov25') }}

    union all

    select
        HHID as household_id,
        county as sub_county,
        cast(null as string) as village_name,
        cast(null as string) as village_code,
        cast(null as string) as farmer_sex,
        cast(null as string) as education_level,
        cast(null as string) as housing_type,
        cast(null as string) as breed_type,
        cast(null as float64) as animals_sold,
        cast(null as float64) as cash_received,
        cast(null as float64) as clinical_disease_score,
        cast(null as float64) as positive_cases,
        cast(null as float64) as incidence_rate,
        Cost_perTreatment as treatment_cost,
        Cattle_number as average_herd_size,
        cast(null as float64) as mortality_count,
        'cattle' as species,
        'ilri_acaricide_use_final_analysis_dataverse' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'ilri_acaricide_use_final_analysis_dataverse') }}

    union all

    select
        q_id as household_id,
        region as sub_county,
        kebele as village_name,
        _village_code as village_code,
        sex_respondent as farmer_sex,
        education_level,
        cast(null as string) as housing_type,
        cast(null as string) as breed_type,
        cast(null as float64) as animals_sold,
        cast(null as float64) as cash_received,
        cast(null as float64) as clinical_disease_score,
        cast(null as float64) as positive_cases,
        cast(null as float64) as incidence_rate,
        cast(null as float64) as treatment_cost,
        safe_cast(size_hh as float64) as average_herd_size,
        cast(null as float64) as mortality_count,
        ls_type as species,
        'ilri_amuse_eth_data' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'ilri_amuse_eth_data') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['household_id', 'source_natural_key', 'species']) }} as ilri_animal_health_sk,
    base.*
from base
