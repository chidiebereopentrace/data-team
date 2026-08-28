{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(f.geography_key, '') || '|' ||
        coalesce(f.item_key, '') || '|' ||
        coalesce(f.forestry_grain, '') || '|' ||
        coalesce(f.unit, '') || '|' ||
        coalesce(cast(f.year as string), '') || '|' ||
        coalesce(f.source_key, '')
    )) as agg_forestry_country_year_key,
    f.geography_key,
    format_date('%Y%m%d', date(f.year, 1, 1)) as date_key,
    g.country_name,
    f.item_key,
    f.forestry_grain,
    f.unit,
    f.year,
    f.source_key,
    avg(f.value) as value_avg,
    count(*) as row_count,
    current_timestamp() as loaded_at
from {{ ref('fct_forestry') }} f
left join {{ ref('dim_geography') }} g
    on g.geography_key = f.geography_key
group by
    f.geography_key,
    g.country_name,
    f.item_key,
    f.forestry_grain,
    f.unit,
    f.year,
    f.source_key
