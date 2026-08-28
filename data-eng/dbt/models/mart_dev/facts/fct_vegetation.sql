{{ config(
    materialized='table',
    cluster_by=['vegetation_grain', 'data_level', 'source_key']
) }}

{# ILRI quantity_* / palatability_* are categorical strings (e.g. Low/High density), not numerics. #}

with ndvi as (
    select
        to_hex(md5(
            'ndvi|' || coalesce(cast(n.objectid as string), '') || '|' ||
            coalesce(n.grid_id, '') || '|' ||
            coalesce(n.source_natural_key, '')
        )) as vegetation_key,
        n.geo_key as geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'n.country_iso3') }} as country_iso3,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        'ndvi_mean' as metric,
        s.source_key as source_id,
        'ndvi_grid' as vegetation_grain,
        n.grid_id,
        n.latitude,
        n.longitude,
        {{ acf_as_of_date('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)', 'n.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
        n.ndvi_mean as value,
        cast(null as string) as unit,
        cast(null as string) as quantity_trees,
        cast(null as string) as quantity_shrubs,
        cast(null as string) as quantity_grass,
        cast(null as string) as palatability_trees,
        cast(null as string) as palatability_shrubs,
        cast(null as string) as palatability_grass,
        cast(null as string) as currently_grazing,
        cast(null as string) as lifeform_mix,
        n.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_ndvi_conformed') }} n
    left join {{ ref('dim_geography') }} g
        on g.geography_key = n.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = n.source_natural_key
),

ilri_scored as (
    select
        v.*,
        case
            when lower(cast(v.quantity_trees as string)) like '%high%' then 3
            when lower(cast(v.quantity_trees as string)) like '%medium%' then 2
            when lower(cast(v.quantity_trees as string)) like '%low%' then 1
            else 0
        end as trees_score,
        case
            when lower(cast(v.quantity_shrubs as string)) like '%high%' then 3
            when lower(cast(v.quantity_shrubs as string)) like '%medium%' then 2
            when lower(cast(v.quantity_shrubs as string)) like '%low%' then 1
            else 0
        end as shrubs_score,
        case
            when lower(cast(v.quantity_grass as string)) like '%high%' then 3
            when lower(cast(v.quantity_grass as string)) like '%medium%' then 2
            when lower(cast(v.quantity_grass as string)) like '%low%' then 1
            else 0
        end as grass_score
    from {{ ref('int_ilri_vegetation_conformed') }} v
),

ilri as (
    select
        to_hex(md5(
            'ilri|' || coalesce(cast(v.latitude as string), '') || '|' ||
            coalesce(cast(v.longitude as string), '') || '|' ||
            coalesce(cast(v.survey_date as string), '') || '|' ||
            coalesce(v.household_id, '') || '|' ||
            coalesce(v.record_type, '') || '|' ||
            coalesce(v.source_natural_key, '')
        )) as vegetation_key,
        v.geo_key as geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'v.country_iso3') }} as country_iso3,
        s.source_key,
        s.tier,
        {{ acf_row_data_level('g.geo_level', 's.default_data_level') }} as data_level,
        {{ acf_geo_scope('g.geo_level', 's.default_data_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        'carrying_capacity' as metric,
        s.source_key as source_id,
        'ilri_site' as vegetation_grain,
        v.household_id as grid_id,
        v.latitude,
        v.longitude,
        {{ acf_as_of_date('safe_cast(v.survey_date as date)', 'cast(null as int64)', 'cast(null as int64)', 'v.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('safe_cast(v.survey_date as date)', 'cast(null as int64)', 'cast(null as int64)') }} as as_of_date_basis,
        safe_cast(v.carrying_capacity as float64) as value,
        cast(null as string) as unit,
        cast(v.quantity_trees as string) as quantity_trees,
        cast(v.quantity_shrubs as string) as quantity_shrubs,
        cast(v.quantity_grass as string) as quantity_grass,
        cast(v.palatability_trees as string) as palatability_trees,
        cast(v.palatability_shrubs as string) as palatability_shrubs,
        cast(v.palatability_grass as string) as palatability_grass,
        cast(v.currently_grazing as string) as currently_grazing,
        case
            when v.grass_score >= v.shrubs_score and v.grass_score >= v.trees_score and v.grass_score > 0
                then 'grass_dominant'
            when v.shrubs_score >= v.trees_score and v.shrubs_score > 0
                then 'shrub_dominant'
            when v.trees_score > 0
                then 'tree_dominant'
        end as lifeform_mix,
        v.source_natural_key,
        current_timestamp() as loaded_at
    from ilri_scored v
    left join {{ ref('dim_geography') }} g
        on g.geography_key = v.geo_key
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = v.source_natural_key
),

base as (
    select * from ndvi
    union all
    select * from ilri
)

select *
from base
qualify row_number() over (
    partition by vegetation_key
    order by value desc nulls last
) = 1
