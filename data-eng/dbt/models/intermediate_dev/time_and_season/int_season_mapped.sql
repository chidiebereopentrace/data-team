{{ config(materialized='table') }}

with by_country as (
    select
        country,
        season_name,
        lower(trim(season_name)) as season_name_norm,
        approx_quantiles(planting_month, 100)[offset(50)] as start_month,
        approx_quantiles(harvest_month, 100)[offset(50)] as end_month,
        count(*) as n_rows
    from {{ ref('stg_yield_raw_data') }}
    where season_name is not null
    group by 1, 2, 3
)

select
    to_hex(md5(season_name_norm)) as season_key,
    to_hex(md5(coalesce(country, '') || '|' || season_name_norm)) as season_country_key,
    country,
    season_name,
    season_name_norm,
    start_month,
    end_month,
    case when end_month < start_month then true else false end as crosses_year,
    case
        when season_name_norm like '%off%'
          or season_name_norm in ('walo', 'bas-fond', 'decrue controlee', 'dam retention')
        then true else false
    end as is_off_season,
    case
        when season_name_norm in ('meher', 'gu', 'deyr', 'short', 'long', 'wet', 'main')
            then 'rains_named'
        when season_name_norm in ('season a', 'season b', 'season c', '1st season', '2nd season', 'first', 'second')
            then 'ordinal'
        when season_name_norm in ('cotton season', 'rice season')
            then 'crop_named'
        when season_name_norm = 'annual'
            then 'annual'
        when season_name_norm in ('walo', 'bas-fond', 'decrue controlee', 'dam retention')
            then 'recession'
        else 'other'
    end as season_family,
    n_rows,
    'yield_raw_data' as source_natural_key,
    current_timestamp() as loaded_at
from by_country
