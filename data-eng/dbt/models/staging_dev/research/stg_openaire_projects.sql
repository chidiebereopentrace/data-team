{{ config(materialized='table') }}

select
    ingestion_id,
    project_id,
    project_code,
    acronym,
    title as project_title,
    start_date,
    currency,
    total_cost,
    funded_amount,
    primary_funder_name,
    jurisdiction,
    funding_stream_desc,
    fetched_at,
    'openaire_projects' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Projects_bronze') }}
