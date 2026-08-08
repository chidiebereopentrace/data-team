{{ config(materialized='table') }}

-- Gold: livestock dimension sourced from FAOSTAT production items that are livestock.
-- Species, breed and production system derived from item names.

with livestock_items as (
    select distinct
        item_code,
        item
    from {{ source('landing', 'FAOstat_africa_Food_Security_and_Nutrition_Suite_of_Food_Security_Indicators') }}
    where lower(item) like '%cattle%'
       or lower(item) like '%sheep%'
       or lower(item) like '%goat%'
       or lower(item) like '%pig%'
       or lower(item) like '%poultry%'
       or lower(item) like '%camel%'
       or lower(item) like '%buffalo%'
       or lower(item) like '%milk%'
       or lower(item) like '%egg%'
)

select
    to_hex(md5(coalesce(item_code, item, '')))     as livestock_key,
    coalesce(item_code, item)                      as livestock_natural_key,
    item                                           as species,
    cast(null as string)                           as breed,
    cast(null as string)                           as production_system,
    cast(null as string)                           as animal_sex,
    cast(null as numeric)                          as exotic_percent,
    cast(null as string)                           as description,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from livestock_items
where item is not null
