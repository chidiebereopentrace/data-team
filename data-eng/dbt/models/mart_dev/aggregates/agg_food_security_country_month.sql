{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(g.geography_key, '') || '|' ||
        coalesce(f.measure_type, '') || '|' ||
        cast(f.year as string) || '|' ||
        cast(f.month as string) || '|' ||
        coalesce(f.source_key, '')
    )) as agg_food_security_country_month_key,
    g.geography_key,
    case
        when f.year is not null and f.month is not null
            then format_date('%Y%m%d', date(f.year, f.month, 1))
    end as date_key,
    g.country_name,
    g.country_iso2,
    f.measure_type,
    f.year,
    f.month,
    f.source_key,
    sum(f.value) as value_sum,
    avg(f.pct_phase3) as pct_phase3_avg,
    avg(f.pct_phase4) as pct_phase4_avg,
    avg(f.pct_phase5) as pct_phase5_avg,
    count(*) as unit_count,
    current_timestamp() as loaded_at
from {{ ref('fct_food_security') }} f
left join {{ ref('dim_geography') }} g
    on g.geography_key = f.geography_key
where f.measure_type = 'population'
group by
    g.geography_key,
    g.country_name,
    g.country_iso2,
    f.measure_type,
    f.year,
    f.month,
    f.source_key
