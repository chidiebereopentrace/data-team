{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(a.geo_key, '') || '|' ||
        coalesce(a.aez_code, '') || '|' ||
        coalesce(cast(a.aez_version as string), '')
    )) as geography_aez_bridge_key,
    a.geo_key as geography_key,
    d.aez_key,
    a.aez_code,
    a.aez_name,
    a.aez_version,
    a.aez_source,
    a.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('int_aez_bridge') }} a
left join {{ ref('dim_aez') }} d
    on d.aez_code = a.aez_code
   and (
        d.aez_version = a.aez_version
        or (d.aez_version is null and a.aez_version is null)
   )
where a.geo_key is not null
  and a.aez_code is not null
qualify row_number() over (
    partition by a.geo_key, a.aez_code, a.aez_version
    order by a.aez_name
) = 1
