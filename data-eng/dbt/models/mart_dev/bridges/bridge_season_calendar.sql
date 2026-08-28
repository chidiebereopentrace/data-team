{{ config(materialized='table') }}

-- Gold: maps seasons to the time keys they cover.
-- For each season, generates a time_key for the start year of that season.

select
    s.season_key,
    t.period_key                                   as time_key,
    true                                           as is_typical_window

from {{ ref('gold_dim_season') }} s
cross join {{ ref('dim_period') }} t
where t.period_year between 2000 and extract(year from current_date())
