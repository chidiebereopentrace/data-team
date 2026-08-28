{{ config(materialized='table') }}

select
    to_hex(md5(cast(person_id as string))) as person_key,
    person_id,
    given_name,
    family_name,
    biography,
    coauthor_count,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('int_openaire_persons') }}
where person_id is not null
qualify row_number() over (
    partition by person_id
    order by fetched_at desc nulls last
) = 1
