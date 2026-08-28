{{ config(
    materialized='table',
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

select
    to_hex(md5(
        coalesce(h.source_natural_key, '') || '|' || coalesce(h.household_id, '')
    )) as household_fact_key,
    hh.household_key,
    g.geography_key,
    g.geo_level,
    {{ acf_country_iso3('g.country_iso3', 'h.country_iso3') }} as country_iso3,
    s.source_key,
    s.tier,
    {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
    {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
    {{ acf_place_scope('g') }} as place_scope,
    'household_snapshot' as metric,
    s.source_key as source_id,
    {{ acf_as_of_date('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)', 'h.loaded_at') }} as as_of_date,
    {{ acf_as_of_date_basis('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
    cast(null as float64) as value,
    cast(null as string) as unit,
    h.hh_size,
    h.hh_size_mae,
    h.land_cultivated,
    h.livestock_holdings,
    h.fies_score,
    h.total_income,
    h.farm_income,
    h.offfarm_income,
    h.food_availability,
    h.food_self_sufficiency,
    h.total_energy_available,
    h.gender_male_control,
    h.gender_female_control,
    h.gender_male_youth_control,
    h.gender_female_youth_control,
    h.crop_diversity,
    h.livestock_diversity,
    h.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('int_ilri_household_conformed') }} h
left join {{ ref('dim_household') }} hh
    on hh.household_id = h.household_id
   and hh.source_natural_key = h.source_natural_key
left join {{ ref('dim_geography') }} g
    on g.geography_key = h.geo_key
left join {{ ref('dim_source') }} s
    on s.source_natural_key = h.source_natural_key
