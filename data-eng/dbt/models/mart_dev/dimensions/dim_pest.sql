{{ config(materialized='table') }}

-- Gold: pest dimension sourced from FAOSTAT pesticides and crop protection data.

with pest_seed as (
    select 'Desert Locust'                         as pest_name, 'Insect'  as pest_type, 'Crops'     as target_crop_livestock
    union all select 'Fall Armyworm',                             'Insect',              'Maize'
    union all select 'Stem Borer',                                'Insect',              'Cereals'
    union all select 'Aphid',                                     'Insect',              'Legumes'
    union all select 'Whitefly',                                  'Insect',              'Vegetables'
    union all select 'Tick',                                      'Arachnid',            'Livestock'
    union all select 'Tsetse Fly',                                'Insect',              'Livestock'
    union all select 'Striga',                                    'Weed',                'Sorghum/Maize'
    union all select 'Fusarium Wilt',                             'Fungal Pathogen',     'Banana'
    union all select 'Coffee Berry Borer',                        'Insect',              'Coffee'
)

select
    to_hex(md5(coalesce(pest_name, '')))           as pest_key,
    to_hex(md5(coalesce(pest_name, '')))           as pest_natural_key,
    pest_name,
    pest_type,
    target_crop_livestock,
    cast(null as string)                           as description,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from pest_seed
