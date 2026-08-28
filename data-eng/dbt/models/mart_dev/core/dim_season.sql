{{ config(materialized='table') }}

select
    season_country_key as season_key,
    country,
    season_name,
    season_name_norm,
    start_month,
    end_month,
    crosses_year,
    is_off_season,
    n_rows,
    current_timestamp() as loaded_at
from {{ ref('int_season_mapped') }}
