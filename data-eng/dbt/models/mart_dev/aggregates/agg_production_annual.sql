{{ config(materialized='table') }}

-- Gold aggregate: annual production summary by country and product.
-- Pre-aggregated for dashboard performance (Power BI / Looker).

select
    f.geo_key,
    g.country_code,
    g.country_name,
    f.time_key,
    f.product_key,
    p.product_name,
    f.source_key,
    src.source_name,
    count(*)                                       as record_count,
    sum(f.production_qty)                          as total_production_qty,
    avg(f.production_qty)                          as avg_production_qty,
    sum(f.area)                                    as total_area,
    avg(f.yield)                                   as avg_yield

from {{ ref('fct_production') }} f
left join {{ ref('gold_dim_geography') }}  g   on f.geo_key = g.geography_key
left join {{ ref('dim_product') }}    p   on f.product_key = p.product_key
left join {{ ref('gold_dim_source') }}     src on f.source_key = src.source_key

group by 1, 2, 3, 4, 5, 6, 7, 8
