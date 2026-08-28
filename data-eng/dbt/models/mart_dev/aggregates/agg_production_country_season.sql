{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(g.geography_key, '') || '|' ||
        coalesce(p.season_key, '') || '|' ||
        coalesce(p.product_key, '') || '|' ||
        cast(p.year as string) || '|' ||
        coalesce(p.source_key, '')
    )) as agg_production_country_season_key,
    g.geography_key,
    p.season_key,
    case
        when p.year is not null then format_date('%Y%m%d', date(p.year, 1, 1))
    end as date_key,
    g.country_name,
    p.product_key,
    p.year as harvest_year,
    p.source_key,
    sum(p.area_harvested) as area_harvested,
    sum(p.production_qty) as production_qty,
    safe_divide(sum(p.production_qty), sum(p.area_harvested)) as yield_recomputed,
    count(*) as fnid_count,
    current_timestamp() as loaded_at
from {{ ref('fct_yield') }} p
left join {{ ref('dim_geography') }} g
    on g.geography_key = p.geography_key
group by
    g.geography_key,
    g.country_name,
    p.season_key,
    p.product_key,
    p.year,
    p.source_key
