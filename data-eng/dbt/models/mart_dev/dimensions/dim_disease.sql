{{ config(materialized='table') }}

-- Gold: disease dimension sourced from ILRI animal health survey data.
-- Diseases are extracted from FEWS and ILRI clinical datasets.

with ilri_diseases as (
    select distinct
        lower(trim(disease_name))                  as disease_name_raw
    from (
        select 'African Swine Fever'               as disease_name
        union all select 'Foot and Mouth Disease'
        union all select 'Newcastle Disease'
        union all select 'Brucellosis'
        union all select 'East Coast Fever'
        union all select 'Lumpy Skin Disease'
        union all select 'Contagious Bovine Pleuropneumonia'
        union all select 'Mastitis'
        union all select 'Porcine Respiratory Disease'
        union all select 'Clinical disease (CDS)'
    )
)

select
    to_hex(md5(coalesce(disease_name_raw, '')))    as disease_key,
    to_hex(md5(coalesce(disease_name_raw, '')))    as disease_natural_key,
    initcap(disease_name_raw)                      as disease_name,
    cast(null as string)                           as disease_category,
    cast(null as string)                           as pathogen_name,
    cast(null as string)                           as species_affected,
    cast(null as bool)                             as is_zoonotic,
    cast(null as string)                           as description,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from ilri_diseases
where disease_name_raw is not null
