{{ config(materialized='table') }}

select
    country_name,
    country_iso2,
    country_iso3,
    population,
    in_africa_scope,
    m49_code,
    current_timestamp() as loaded_at
from {{ ref('int_ref_country') }}
