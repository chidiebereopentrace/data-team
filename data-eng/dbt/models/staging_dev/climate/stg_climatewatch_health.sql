{{ config(materialized='table') }}

with base as (
    select
        location as country_name,
        iso_code2 as country_code,
        model,
        scenario,
        category,
        subcategory,
        indicator,
        unit,
        year,
        value,
        'ClimateWatch_climate_health_impacts' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'climatewatch_climate_health_impacts') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['country_code', 'model', 'scenario', 'category', 'subcategory', 'indicator', 'year']) }} as climatewatch_health_sk,
    base.*
from base
