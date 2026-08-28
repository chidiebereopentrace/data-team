{{ config(materialized='table') }}

-- Gold: water source dimension for household water access classification.

with water_sources as (
    select 'PIPED_IN'    as water_source, 'Piped (indoors)'     as water_access_type, 'Piped water directly into dwelling'  as description
    union all select 'PIPED_YARD',        'Piped (yard)',         'Piped water to yard/plot'
    union all select 'BOREHOLE',          'Borehole',             'Protected borehole or tubewell'
    union all select 'PROTECTED_SPRING',  'Protected Spring',     'Protected spring water source'
    union all select 'UNPROTECTED_WELL',  'Unprotected Well',     'Unprotected dug well'
    union all select 'SURFACE_WATER',     'Surface Water',        'River, lake, or pond water'
    union all select 'RAINWATER',         'Rainwater Collection', 'Rainwater harvesting'
    union all select 'TANKER',            'Tanker/Truck',         'Water delivered by tanker or truck'
)

select
    to_hex(md5(coalesce(water_source, '')))        as water_key,
    water_source,
    water_access_type,
    description

from water_sources
