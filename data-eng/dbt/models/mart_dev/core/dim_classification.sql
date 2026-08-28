{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(phase_code, '') || '|' || coalesce(phase_name, '')
    )) as classification_key,
    phase_code,
    phase_name,
    any_value(classification_scale) as classification_scale,
    current_timestamp() as loaded_at
from {{ ref('int_fews_food_security_conformed') }}
where phase_code is not null or phase_name is not null
group by
    phase_code,
    phase_name
