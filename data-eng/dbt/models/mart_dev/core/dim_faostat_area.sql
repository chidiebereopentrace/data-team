{{ config(materialized='table') }}

select
    r.area_code as faostat_area_key,
    r.area_code,
    r.area_code_m49,
    r.country_name,
    r.country_iso2,
    r.country_iso3,
    r.match_method,
    coalesce(a.area_code is not null, false) as is_aggregate_area,
    a.reason as aggregate_reason,
    current_timestamp() as loaded_at
from {{ ref('int_faostat_area_reference') }} r
left join {{ ref('ref_faostat_aggregate_areas') }} a
    on cast(a.area_code as string) = cast(r.area_code as string)
