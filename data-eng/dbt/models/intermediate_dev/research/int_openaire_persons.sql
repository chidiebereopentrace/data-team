{{ config(materialized='table') }}

select
    person_id,
    given_name,
    family_name,
    biography,
    coauthor_count,
    fetched_at,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_openaire_persons') }}
where person_id is not null
