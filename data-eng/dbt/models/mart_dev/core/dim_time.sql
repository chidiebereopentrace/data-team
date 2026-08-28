{{ config(materialized='table') }}

-- Gold: full time dimension extended from staging_dev dim_period.
-- Derives quarter, month, week, and weekend flag from the period year.

select
    cast(period_key as int64)                           as time_key,
    start_date                                          as date_actual,
    period_year                                         as year,
    cast(ceil(extract(month from start_date) / 3.0) as int64)
                                                        as quarter,
    extract(month from start_date)                      as month,
    format_date('%B', start_date)                       as month_name,
    extract(day from start_date)                        as day_of_month,
    extract(dayofweek from start_date)                  as day_of_week,
    extract(week from start_date)                       as week_of_year,
    extract(dayofweek from start_date) in (1, 7)        as is_weekend,
    period_year                                         as fiscal_year,
    current_timestamp()                                 as created_at

from {{ ref('dim_period') }}
