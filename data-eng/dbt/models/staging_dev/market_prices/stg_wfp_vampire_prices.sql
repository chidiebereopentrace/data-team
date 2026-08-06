{{ config(materialized='table') }}

select
    adm0_id as country_id,
    adm0_name as country,
    adm1_id as admin1_id,
    adm1_name as admin1_name,
    mkt_id as market_id,
    mkt_name as market_name,
    cm_id as product_id,
    cm_name as product_name,
    cur_id as currency_id,
    cur_name as currency,
    pt_id as price_type_id,
    pt_name as price_type,
    um_id as unit_id,
    um_name as unit,
    mp_year as year,
    mp_month as month,
    mp_price as value,
    mp_commoditysource as commodity_source,
    'WFP_VAMPIRE_global_food_prices' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'WFP_VAMPIRE_Tool_global_food_prices_bronze') }}
