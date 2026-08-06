{{ config(materialized='table') }}

select
    person_id,
    given_name,
    family_name,
    biography,
    coauthor_count,
    fetched_at,
    'openaire_persons' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Persons_bronze') }}
