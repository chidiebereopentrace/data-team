{{ config(materialized='table') }}

with keys as (
    select source_natural_key from {{ ref('stg_fews_food_security') }}
    union distinct select source_natural_key from {{ ref('stg_fews_market_prices') }}
    union distinct select source_natural_key from {{ ref('stg_fews_cross_border_trade') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_production') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_macro') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_prices') }}
    union distinct select source_natural_key from {{ ref('stg_wfp_vampire_prices') }}
    union distinct select source_natural_key from {{ ref('stg_yield_raw_data') }}
    union distinct select source_natural_key from {{ ref('stg_africa_hdi') }}
    union distinct select source_natural_key from {{ ref('stg_africa_gdp_ppp') }}
    union distinct select source_natural_key from {{ ref('stg_isda_soil_enriched') }}
    union distinct select source_natural_key from {{ ref('stg_isric_africa_soil') }}
)

select
    source_natural_key,
    source_natural_key as source_id,
    case
        when source_natural_key like 'FEWS%' then 'FEWS NET'
        when source_natural_key like 'Unpivoted_FAOstat%' then 'FAOSTAT'
        when lower(source_natural_key) like '%vampire%' or source_natural_key like 'WFP%' then 'WFP'
        when source_natural_key like 'ilri%' then 'ILRI'
        when lower(source_natural_key) like '%isric%' then 'ISRIC'
        when lower(source_natural_key) like '%isda%' then 'iSDA'
        when source_natural_key like '%Human_development%' then 'UNDP / HDI extract'
        when source_natural_key like '%gdp%' or source_natural_key like '%GDP%' then 'World Bank / GDP extract'
        when source_natural_key = 'yield_raw_data' then 'Yield extract'
        else 'other'
    end as organisation_name,
    case
        when source_natural_key like 'FEWS%' then 1
        when source_natural_key like 'ilri%' then 3
        when lower(source_natural_key) like '%isric%' or lower(source_natural_key) like '%isda%' then 3
        else 2
    end as tier,
    case
        when source_natural_key like 'FEWS%' then 'sub_national'
        when source_natural_key like 'ilri%' then 'community'
        when lower(source_natural_key) like '%isric%' or lower(source_natural_key) like '%isda%' then 'community'
        else 'national'
    end as data_level,
    cast(null as date) as as_of_date,
    cast(null as string) as region,
    current_timestamp() as loaded_at
from keys
where source_natural_key is not null
