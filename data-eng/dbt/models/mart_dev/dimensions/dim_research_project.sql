{{ config(materialized='table') }}

-- Gold: research project dimension sourced from OpenAire projects.

-- Actual columns: project_id, project_code, acronym, title, start_date,
--                  currency, total_cost, funded_amount, primary_funder_name,
--                  jurisdiction, funding_stream_desc, ingestion_id, fetched_at

select
    to_hex(md5(coalesce(project_id, title, '')))  as research_project_key,
    coalesce(project_code, project_id)             as project_code,
    title                                          as project_name,
    cast(null as string)                           as doi,
    cast(null as string)                           as journal_name,
    cast(null as string)                           as keywords,
    cast(start_date as date)                       as start_date,
    cast(null as date)                             as end_date,
    funding_stream_desc                            as description

from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Projects_bronze') }}
where coalesce(project_id, title) is not null
