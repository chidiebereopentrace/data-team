{{ config(materialized='table') }}

-- Gold: breed and variety dimension sourced from ARCGIS germplasm and crop data.

with germplasm as (
    select distinct
        cast(null as string)                       as variety_code,
        taxon                                      as variety_name,
        'crop'                                     as species
    from {{ source('raw_dev', 'arcgis_layer_rice_germplasm_in_africa_3d2a9') }}
    where taxon is not null
)

select
    to_hex(md5(concat(
        coalesce(variety_name, ''), '|', coalesce(species, '')
    )))                                            as breed_variety_key,
    variety_code,
    variety_name,
    species,
    cast(null as string)                           as description,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from germplasm
