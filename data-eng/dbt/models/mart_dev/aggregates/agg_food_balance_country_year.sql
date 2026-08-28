{{ config(materialized='table') }}

{# Aggregate convenience metrics from long-form element rows (by element_code). #}

select
    to_hex(md5(
        coalesce(f.geography_key, '') || '|' ||
        coalesce(f.product_key, '') || '|' ||
        coalesce(cast(f.year as string), '') || '|' ||
        coalesce(f.source_key, '')
    )) as agg_food_balance_country_year_key,
    f.geography_key,
    format_date('%Y%m%d', date(f.year, 1, 1)) as date_key,
    g.country_name,
    f.product_key,
    f.year,
    f.source_key,
    sum(f.production) as production_sum,
    sum(f.imports) as imports_sum,
    sum(f.exports) as exports_sum,
    sum(f.food) as food_sum,
    sum(f.feed) as feed_sum,
    sum(f.losses) as losses_sum,
    count(*) as row_count,
    current_timestamp() as loaded_at
from {{ ref('fct_food_balance') }} f
left join {{ ref('dim_geography') }} g
    on g.geography_key = f.geography_key
group by
    f.geography_key,
    g.country_name,
    f.product_key,
    f.year,
    f.source_key
