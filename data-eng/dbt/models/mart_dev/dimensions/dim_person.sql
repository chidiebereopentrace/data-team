{{ config(materialized='table') }}

-- Gold: person dimension sourced from OpenAire research persons.
-- Covers researchers and authors; household-level persons added as ILRI data matures.

select
    to_hex(md5(coalesce(person_id, '')))           as person_key,
    coalesce(person_id, '')                        as person_natural_key,
    concat(coalesce(given_name, ''), ' ', coalesce(family_name, '')) as full_name,
    cast(null as string)                           as gender,
    cast(null as int64)                            as age,
    cast(null as int64)                            as year_of_birth,
    cast(null as string)                           as education_level,
    cast(null as string)                           as marital_status,
    cast(null as int64)                            as years_farm_experience,
    'researcher'                                   as role,
    cast(null as string)                           as household_id,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Persons_bronze') }}
-- source confirmed in sources_raw_dev.yml
where person_id is not null
