{{ config(materialized='table') }}

-- Gold: vegetation fact from ARCGIS NDVI satellite data.

select
    to_hex(md5(concat(
        coalesce(cast(objectid as string), ''), '|',
        coalesce(grid_id, '')
    )))                                            as vegetation_key,

    to_hex(md5(concat(
        coalesce(cast(objectid as string), ''), '|',
        coalesce(grid_id, ''), '|', ''
    )))                                            as geo_key,

    cast(null as int64)                            as time_key,
    to_hex(md5('arcgis_vegetation_ndvi'))          as source_key,
    cast(null as string)                           as unit_key,
    cast(null as string)                           as audit_key,

    cast(null as numeric)                          as quantity_trees,
    cast(null as numeric)                          as quantity_shrubs,
    cast(null as numeric)                          as quantity_grass,
    cast(null as numeric)                          as carrying_capacity,
    cast(mean as numeric)                          as ndvi,
    cast(null as string)                           as leaves_condition_trees,
    cast(null as string)                           as leaves_condition_shrubs,
    cast(null as string)                           as leaves_condition_grass,
    cast(null as string)                           as palatability_trees,
    cast(null as string)                           as palatability_shrubs,
    cast(objectid as string)                       as source_record_id

from {{ source('raw_dev', 'arcgis_vegetation_ndvi') }}
where mean is not null
