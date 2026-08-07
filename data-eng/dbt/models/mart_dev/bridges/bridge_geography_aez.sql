{{ config(materialized='table') }}

-- Gold: links geography keys to AEZ (Agro-Ecological Zone) codes.
-- AEZ attributes are embedded in FAOSTAT and IFPRI spatial references.

with aez_ref as (
    select distinct
        to_hex(md5(concat(
            coalesce(cast(area_code_m49 as string), ''), '|',
            coalesce(area, ''), '|', ''
        )))                                        as geo_key,
        cast(null as string)                       as aez_code,
        cast(null as string)                       as aez_name,
        '2022'                                     as aez_version,
        'IFPRI'                                    as aez_source,
        cast(null as numeric)                      as aez_share
    from {{ source('landing', 'FAOstat_africa_Food_Security_and_Nutrition_Suite_of_Food_Security_Indicators') }}
    where area_code_m49 is not null
      and area is not null
)

select
    geo_key,
    aez_code,
    aez_name,
    aez_version,
    aez_source,
    aez_share

from aez_ref
where aez_code is not null
