select
    count(distinct measure_type) as measure_type_count
from {{ ref('stg_fews_food_security') }}
having count(distinct measure_type) < 2
