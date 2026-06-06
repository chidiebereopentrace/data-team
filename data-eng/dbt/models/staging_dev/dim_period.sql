{{ config(materialized='table') }}

select distinct
    cast(period_year as int64) as period_key,
    cast(period_year as int64) as period_year,
    date(cast(period_year as int64), 1, 1) as start_date
from {{ ref('stg_silver_star_metrics') }}
where period_year is not null
