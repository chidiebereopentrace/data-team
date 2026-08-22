{{ config(materialized='table') }}

-- ILRI Household Food Security: CRP survey indicators + Indicators Final.

with base as (
    select
        hh_id as household_id,
        country,
        region,
        district,
        village_code,
        household_size_members as hh_size,
        household_size_mae as hh_size_mae,
        household_type,
        head_education_level,
        land_cultivated,
        livestock_holdings,
        fies_score,
        total_income,
        farm_income,
        offfarm_income,
        value_farm_produce,
        crop_sales,
        value_crop_produce,
        value_crop_consumed,
        livestock_product_sales,
        value_livestock_production,
        value_livestock_consumed,
        food_availability,
        food_self_sufficiency,
        total_energy_available,
        gender_male_control,
        gender_female_control,
        gender_male_youth_control,
        gender_female_youth_control,
        crop_diversity,
        livestock_diversity,
        'ilri_crp_household_food_security_v1' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'ilri_crp_household_food_security_v1') }}

    union all

    select
        HHid as household_id,
        Country as country,
        Region as region,
        district,
        cast(Village as string) as village_code,
        cast(HHsizemembers as int64) as hh_size,
        HHsizeMAE as hh_size_mae,
        HouseholdType as household_type,
        Head_EducationLevel as head_education_level,
        LandCultivated as land_cultivated,
        LivestockHoldings as livestock_holdings,
        cast(FIES_Score as float64) as fies_score,
        total_income,
        farm_income,
        offfarm_income,
        valuefarmproduce as value_farm_produce,
        cropsales as crop_sales,
        valuecropproduce as value_crop_produce,
        cast(null as float64) as value_crop_consumed,
        cast(null as float64) as livestock_product_sales,
        cast(null as float64) as value_livestock_production,
        cast(null as float64) as value_livestock_consumed,
        cast(null as float64) as food_availability,
        cast(null as float64) as food_self_sufficiency,
        cast(null as float64) as total_energy_available,
        cast(null as float64) as gender_male_control,
        cast(null as float64) as gender_female_control,
        cast(null as float64) as gender_male_youth_control,
        cast(null as float64) as gender_female_youth_control,
        cast(null as float64) as crop_diversity,
        cast(null as float64) as livestock_diversity,
        'ilri_indicators_final_v1' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'ilri_indicators_final_v1') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['household_id', 'source_natural_key']) }} as ilri_household_food_security_sk,
    base.*
from base
