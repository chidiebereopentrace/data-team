{{ config(materialized='table') }}

select
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
    fetched_at,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_openaire_projects') }}
where project_id is not null
