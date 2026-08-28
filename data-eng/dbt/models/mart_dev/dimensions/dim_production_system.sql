{{ config(materialized='table') }}

-- Gold: production system dimension for livestock and crop systems.

with systems as (
    select 'RAIN'   as system_code, 'Rainfed Smallholder'          as system_name, 'Rain-dependent smallholder farming'           as description
    union all select 'IRR',         'Irrigated',                    'Irrigated crop/livestock systems'
    union all select 'PAST',        'Pastoral',                     'Nomadic/transhumant pastoral systems'
    union all select 'AGRO',        'Agro-Pastoral',                'Mixed crop-livestock agro-pastoral'
    union all select 'COMM',        'Commercial',                   'Large-scale commercial farming'
    union all select 'MIXED',       'Mixed Crop-Livestock',         'Integrated smallholder crop-livestock'
    union all select 'PERI',        'Peri-Urban',                   'Peri-urban market-oriented systems'
    union all select 'FOREST',      'Forest-Based',                 'Agroforestry and forest-adjacent systems'
)

select
    to_hex(md5(coalesce(system_code, '')))         as production_system_key,
    system_code,
    system_name,
    description

from systems
