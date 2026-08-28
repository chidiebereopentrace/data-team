{{ config(materialized='table') }}

with {{ geo_country_by_iso2_cte('iso2_geo') }},

admin1_geo as (
    select
        upper(trim(country_iso2)) as country_iso2,
        geo_key,
        country_iso3,
        country_name,
        admin_1_name,
        replace(lower(trim(admin_1_name)), ' ', '') as admin_1_norm
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'admin1'
      and admin_1_name is not null
      and country_iso2 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso2)), replace(lower(trim(admin_1_name)), ' ', '')
        order by population desc nulls last, geo_key
    ) = 1
)

select
    a.household_id,
    a.sub_county,
    a.village_name,
    a.village_code,
    a.farmer_sex,
    a.education_level,
    a.housing_type,
    a.breed_type,
    a.animals_sold,
    a.cash_received,
    a.clinical_disease_score,
    a.positive_cases,
    a.incidence_rate,
    a.treatment_cost,
    a.average_herd_size,
    a.mortality_count,
    a.species,
    a.country_iso2,
    coalesce(g_admin.country_iso3, g_country.country_iso3, a.country_iso3) as country_iso3,
    coalesce(g_admin.geo_key, g_country.geo_key) as geo_key,
    case
        when g_admin.geo_key is not null then 'admin1'
        when g_country.geo_key is not null then 'country'
    end as geo_level,
    a.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_ilri_animal_health') }} a
left join admin1_geo g_admin
    on g_admin.country_iso2 = a.country_iso2
   and a.sub_county is not null
   and g_admin.admin_1_norm = replace(lower(trim(a.sub_county)), ' ', '')
left join iso2_geo g_country
    on g_country.country_iso2 = a.country_iso2
where a.household_id is not null
