{{ config(materialized='table') }}

-- Gold: product/crop dimension extended from staging_dev dim_crop.
-- Covers crops and food products; livestock handled separately in dim_livestock.

select
    to_hex(md5(coalesce(crop_name, '')))           as product_key,
    to_hex(md5(coalesce(crop_name, '')))           as product_natural_key,
    crop_name                                      as product_name,
    cast(null as string)                           as cpcv2,
    cast(null as string)                           as cpcv2_description,
    cast(null as string)                           as scientific_name,
    cast(null as string)                           as aliases,
    cast(null as bool)                             as is_staple_food,
    'crop'                                         as product_source,
    cast(null as string)                           as food_group_code,
    cast(null as string)                           as food_group,
    cast(null as string)                           as nutrient_type,
    cast(null as numeric)                          as dry_matter_pct,
    cast(null as numeric)                          as crude_protein_pct,
    cast(null as numeric)                          as gross_energy_mj_kg,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from {{ ref('dim_crop') }}
