{{ config(materialized='table') }}

select
    to_hex(md5(cast(project_id as string))) as research_project_key,
    project_id,
    project_code,
    acronym,
    project_title,
    start_date,
    currency,
    total_cost,
    funded_amount,
    primary_funder_name,
    jurisdiction,
    funding_stream_desc,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('int_openaire_projects') }}
where project_id is not null
qualify row_number() over (
    partition by project_id
    order by fetched_at desc nulls last
) = 1
