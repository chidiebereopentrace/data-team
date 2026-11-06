{{ config(materialized='table') }}

{% set iso3_codes = [
    'ago', 'bdi', 'ben', 'bfa', 'caf', 'civ', 'cmr', 'cod', 'cog', 'cpv',
    'dji', 'dza', 'egy', 'eri', 'eth', 'gab', 'gha', 'gin', 'gmb', 'gnb',
    'ken', 'lbr', 'lby', 'lso', 'mdg', 'mli', 'moz', 'mrt', 'mwi', 'nam',
    'ner', 'nga', 'rwa', 'sdn', 'sen', 'sle', 'som', 'ssd', 'swz', 'tcd',
    'tgo', 'tza', 'uga', 'zaf', 'zmb', 'zwe'
] %}

with prices as (
    {% for iso in iso3_codes %}
    select
        country_name as country,
        country_iso3 as country_code,
        admin1 as admin_1,
        admin2 as admin_2,
        market as market_name,
        commodity as product_name,
        pricetype as price_type,
        unit,
        currency,
        price,
        extract(year from date) as year,
        extract(month from date) as month
    from {{ source('raw_dev', 'hdx_wfp_food_prices_' ~ iso) }}
    where price is not null
      and date is not null
      and country_name is not null
      and market is not null
      and commodity is not null
    {% if not loop.last %}
    union all
    {% endif %}
    {% endfor %}
)

select
    country,
    country_code,
    admin_1,
    admin_2,
    market_name,
    product_name,
    price_type,
    unit,
    currency,
    avg(price) as value,
    year,
    month,
    'WFP_HDX_country_food_prices' as source_natural_key,
    current_timestamp() as loaded_at
from prices
where year is not null
  and month between 1 and 12
group by
    country,
    country_code,
    admin_1,
    admin_2,
    market_name,
    product_name,
    price_type,
    unit,
    currency,
    year,
    month
