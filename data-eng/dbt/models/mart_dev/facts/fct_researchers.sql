{{ config(materialized='table') }}

-- Gold: researchers fact from OpenAire persons (research publications).

-- Actual columns: person_id, given_name, family_name, biography, coauthor_count, fetched_at

select
    to_hex(md5(coalesce(person_id, '')))           as researcher_fact_key,

    cast(null as string)                           as geo_key,

    cast(null as int64)                            as time_key,
    to_hex(md5('openaire_agriculture_and_environment_Research_publications_Persons_bronze'))
                                                   as source_key,
    cast(null as string)                           as organisation_key,
    to_hex(md5(coalesce(person_id, '')))           as person_key,
    cast(null as string)                           as unit_key,
    cast(null as numeric)                          as researcher_count_fte,
    cast(null as int64)                            as researcher_count_hc,
    coalesce(person_id, '')                        as source_record_id

from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Persons_bronze') }}
where person_id is not null
