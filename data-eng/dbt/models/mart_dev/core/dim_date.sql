{{ config(materialized='table') }}

select
    format_date('%Y%m%d', d) as date_key,
    d as date_day,
    extract(year from d) as year,
    extract(month from d) as month,
    extract(quarter from d) as quarter,
    format_date('%Y-%m', d) as year_month
from unnest(generate_date_array('1960-01-01', '2035-12-31', interval 1 day)) as d
