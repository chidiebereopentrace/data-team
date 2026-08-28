{{ config(materialized='table') }}

with country_geo as (
    select
        lower(trim(country_name)) as country_name_norm,
        geo_key,
        country_iso3
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'country'
      and country_name is not null
    qualify row_number() over (
        partition by lower(trim(country_name))
        order by population desc nulls last, geo_key
    ) = 1
)

select
    h.household_id,
    h.country,
    h.region,
    h.district,
    h.village_code,
    h.hh_size,
    h.hh_size_mae,
    h.household_type,
    h.head_education_level,
    h.land_cultivated,
    h.livestock_holdings,
    h.fies_score,
    h.total_income,
    h.farm_income,
    h.offfarm_income,
    h.value_farm_produce,
    h.crop_sales,
    h.value_crop_produce,
    h.value_crop_consumed,
    h.livestock_product_sales,
    h.value_livestock_production,
    h.value_livestock_consumed,
    h.food_availability,
    h.food_self_sufficiency,
    h.total_energy_available,
    h.gender_male_control,
    h.gender_female_control,
    h.gender_male_youth_control,
    h.gender_female_youth_control,
    h.crop_diversity,
    h.livestock_diversity,
    g.geo_key,
    g.country_iso3,
    h.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_ilri_household_food_security') }} h
left join country_geo g
    on g.country_name_norm = lower(trim(h.country))
where h.household_id is not null
