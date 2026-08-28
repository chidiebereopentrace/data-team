{{ config(materialized='table') }}

select
    to_hex(md5(
        lower(trim(coalesce(country, ''))) || '|' ||
        lower(trim(coalesce(market_name, '')))
    )) as market_key,
    country,
    admin_1,
    market_name,
    current_timestamp() as loaded_at
from {{ ref('int_prices_harmonised') }}
where market_name is not null
qualify row_number() over (
    partition by country, market_name
    order by admin_1 desc nulls last
) = 1
