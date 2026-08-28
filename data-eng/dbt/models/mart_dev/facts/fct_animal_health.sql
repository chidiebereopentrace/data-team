{{ config(
    materialized='table',
    cluster_by=['source_key', 'country_iso3']
) }}

with base as (
    select
        to_hex(md5(
            coalesce(a.household_id, '') || '|' ||
            coalesce(a.species, '') || '|' ||
            coalesce(a.source_natural_key, '')
        )) as animal_health_key,
        a.geo_key as geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'a.country_iso3') }} as country_iso3,
        hh.household_key,
        lv.livestock_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('coalesce(g.geo_level, a.geo_level)') }} as data_level,
        {{ acf_geo_scope_strict('coalesce(g.geo_level, a.geo_level)') }} as geo_scope,
        array(
            select distinct x
            from unnest([
                coalesce(g.country_iso3, a.country_iso3),
                g.fnid,
                coalesce(g.admin_1_name, a.sub_county),
                coalesce(g.admin_2_name, a.village_name),
                g.city_name,
                coalesce(g.country_name, cast(null as string))
            ]) as x
            where x is not null and x != ''
        ) as place_scope,
        concat('animal_health_', lower(coalesce(a.species, ''))) as metric,
        s.source_key as source_id,
        {{ acf_as_of_date('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)', 'a.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
        cast(a.average_herd_size as float64) as value,
        cast(null as string) as unit,
        a.sub_county,
        a.village_name,
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
        a.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_ilri_animal_health_conformed') }} a
    left join {{ ref('dim_geography') }} g
        on g.geography_key = a.geo_key
    left join {{ ref('dim_household') }} hh
        on hh.household_id = a.household_id
       and hh.source_natural_key = a.source_natural_key
    left join {{ ref('dim_livestock') }} lv
        on lower(lv.species) = lower(a.species)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = a.source_natural_key
)

select *
from base
qualify row_number() over (
    partition by animal_health_key
    order by mortality_count desc nulls last, average_herd_size desc nulls last
) = 1
