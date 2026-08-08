{{ config(materialized='table') }}

-- Gold: land use classification dimension sourced from FAOSTAT land use categories.

with land_use_classes as (
    select distinct
        lower(trim(item))                          as land_use_code,
        item                                       as land_use_class
    from {{ source('raw_dev', 'fao_land_use_bronze') }}
    where item is not null
)

select
    to_hex(md5(coalesce(land_use_code, '')))       as land_use_key,
    land_use_code,
    land_use_class,
    cast(null as string)                           as description

from land_use_classes
