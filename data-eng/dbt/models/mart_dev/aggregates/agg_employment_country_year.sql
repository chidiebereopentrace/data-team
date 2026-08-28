{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(e.geography_key, '') || '|' ||
        coalesce(e.indicator, '') || '|' ||
        coalesce(e.sex, '') || '|' ||
        coalesce(e.unit, '') || '|' ||
        cast(e.year as string) || '|' ||
        coalesce(e.source_key, '')
    )) as agg_employment_country_year_key,
    e.geography_key,
    format_date('%Y%m%d', date(e.year, 1, 1)) as date_key,
    g.country_name,
    e.indicator,
    e.sex,
    e.unit,
    e.year,
    e.source_key,
    avg(e.value) as value_avg,
    count(*) as row_count,
    current_timestamp() as loaded_at
from {{ ref('fct_employment') }} e
left join {{ ref('dim_geography') }} g
    on g.geography_key = e.geography_key
group by
    e.geography_key,
    g.country_name,
    e.indicator,
    e.sex,
    e.unit,
    e.year,
    e.source_key
