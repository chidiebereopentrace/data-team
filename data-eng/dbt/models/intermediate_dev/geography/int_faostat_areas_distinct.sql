{{ config(materialized='table') }}

{# Distinct FAOSTAT area_code rows across all FAOSTAT staging sources. #}

with areas as (
    select distinct
        cast(area_code as string) as area_code,
        {{ geo_faostat_m49_norm('area_code_m49') }} as area_code_m49,
        trim(country_name) as country_name
    from {{ ref('stg_faostat_production') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_population_employment') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_food_balances') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_emissions') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_forestry') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_investment_asti') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_sdg_hdi') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_food_aid') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_land_inputs') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_discontinued_machinery') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_macro') }}
    where country_name is not null

    union distinct

    select distinct
        cast(area_code as string),
        {{ geo_faostat_m49_norm('area_code_m49') }},
        trim(country_name)
    from {{ ref('stg_faostat_trade') }}
    where country_name is not null
)

select
    area_code,
    area_code_m49,
    country_name,
    current_timestamp() as loaded_at
from areas
qualify row_number() over (
    partition by area_code
    order by area_code_m49 nulls last, country_name
) = 1
