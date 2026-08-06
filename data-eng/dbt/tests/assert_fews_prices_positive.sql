select *
from {{ ref('stg_fews_market_prices') }}
where value < 0
