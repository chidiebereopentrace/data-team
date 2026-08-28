{{ config(materialized='table') }}

select
    to_hex(md5(src.hazard_name_norm)) as disease_key,
    any_value(src.hazard_name) as disease_or_hazard,
    'foodborne' as disease_family,
    current_timestamp() as loaded_at
from (
    select foodborne_hazard as hazard_name, lower(trim(foodborne_hazard)) as hazard_name_norm
    from {{ ref('int_food_hazards_conformed') }}
    where foodborne_hazard is not null
) src
group by src.hazard_name_norm
