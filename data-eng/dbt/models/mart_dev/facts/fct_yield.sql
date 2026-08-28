{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key']
) }}

{# FNID–season yield (yield_raw_data). FAOSTAT country–year lives in fct_production. #}

with base as (
    select
        to_hex(md5(
            'yield|' || coalesce(y.fnid, '') || '|' ||
            coalesce(y.product, '') || '|' ||
            coalesce(y.season_name, '') || '|' ||
            cast(y.harvest_year as string) || '|' ||
            cast(y.harvest_month as string) || '|' ||
            coalesce(y.crop_production_system, '') || '|' ||
            y.source_natural_key
        )) as yield_key,
        g.geography_key,
        g.geo_level,
        {{ acf_country_iso3('g.country_iso3', 'y.country_iso3') }} as country_iso3,
        se.season_key,
        pr.product_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('g.geo_level') }} as data_level,
        {{ acf_geo_scope_strict('g.geo_level') }} as geo_scope,
        {{ acf_place_scope('g') }} as place_scope,
        concat('production_', lower(replace(coalesce(y.product, ''), ' ', '_')), '_yield') as metric,
        s.source_key as source_id,
        u.unit_key as production_unit_key,
        y.harvest_year as year,
        y.harvest_month as month,
        case
            when y.harvest_year is not null and y.harvest_month between 1 and 12
                then format_date('%Y%m%d', date(y.harvest_year, y.harvest_month, 1))
            when y.harvest_year is not null
                then format_date('%Y%m%d', date(y.harvest_year, 12, 31))
        end as date_key,
        {{ acf_as_of_date('cast(null as date)', 'y.harvest_year', 'y.harvest_month', 'y.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'y.harvest_year', 'y.harvest_month') }} as as_of_date_basis,
        'fnid_season' as grain_flag,
        y.area as area_harvested,
        y.production as production_qty,
        y.yield as yield_value,
        'ha' as area_unit,
        'tonnes' as production_unit,
        't/ha' as yield_unit,
        coalesce(y.production, y.yield) as value,
        y.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_yield_raw_with_geo') }} y
    left join {{ ref('dim_geography') }} g
        on g.geography_key = y.geo_key
    left join {{ ref('dim_season') }} se
        on se.country = y.country
       and se.season_name_norm = lower(trim(y.season_name))
    left join {{ ref('dim_product') }} pr
        on pr.item_code is null
       and lower(pr.product_name) = lower(y.product)
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = y.source_natural_key
    left join {{ ref('dim_unit') }} u
        on lower(u.unit_code) = 'tonnes'
       and u.unit_type = 'quantity'
)

select *
from base
qualify row_number() over (
    partition by yield_key
    order by production_qty desc nulls last, yield_value desc nulls last
) = 1
