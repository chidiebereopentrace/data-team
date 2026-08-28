{{ config(materialized='table') }}

select
    to_hex(md5(src.species_norm)) as livestock_key,
    any_value(src.species) as species,
    current_timestamp() as loaded_at
from (
    select species, lower(trim(species)) as species_norm
    from {{ ref('int_ilri_animal_health_conformed') }}
    where species is not null
) src
group by src.species_norm
