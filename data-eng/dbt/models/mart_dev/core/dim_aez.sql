{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(a.aez_code, '') || '|' ||
        coalesce(cast(a.aez_version as string), '')
    )) as aez_key,
    any_value(a.aez_code) as aez_code,
    any_value(a.aez_name) as aez_name,
    a.aez_version,
    any_value(a.aez_source) as aez_source,
    current_timestamp() as loaded_at
from {{ ref('int_aez_bridge') }} a
where a.aez_code is not null
group by
    a.aez_code,
    a.aez_version
