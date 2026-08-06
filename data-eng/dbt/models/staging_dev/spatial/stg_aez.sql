{{ config(materialized='table') }}

-- Schema placeholder until AEZ zonal stats are loaded into raw_dev.

select
    cast(null as string) as geo_key,
    cast(null as string) as aez_code,
    cast(null as string) as aez_name,
    cast(null as string) as aez_version,
    cast(null as string) as aez_source,
    'aez_zonal_stats' as source_natural_key,
    current_timestamp() as loaded_at
from (select 1 as _x)
where false
