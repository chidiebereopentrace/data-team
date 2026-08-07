{{ config(materialized='table') }}

-- Gold: protected areas fact from ARCGIS land protected areas dataset.

select
    to_hex(md5(concat(
        coalesce(cast(objectid as string), ''), '|',
        coalesce(name, '')
    )))                                            as protected_area_key,

    to_hex(md5(concat(
        coalesce(cast(objectid as string), ''), '|',
        coalesce(name, ''), '|', ''
    )))                                            as geo_key,

    cast(null as int64)                            as time_key,
    to_hex(md5('arcgis_land_protected_areas'))     as source_key,
    cast(analysisarea as numeric)                  as area_protected,
    cast(null as numeric)                          as protection_level,
    name                                           as designation,
    cast(objectid as string)                       as source_record_id

from {{ source('raw_dev', 'arcgis_land_protected_areas') }}
where objectid is not null
