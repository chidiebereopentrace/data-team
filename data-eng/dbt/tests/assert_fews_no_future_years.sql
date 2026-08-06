select 'stg_fews_food_security' as model_name, year
from {{ ref('stg_fews_food_security') }}
where year > extract(year from current_date())

union all

select 'stg_fews_market_prices' as model_name, year
from {{ ref('stg_fews_market_prices') }}
where year > extract(year from current_date())
