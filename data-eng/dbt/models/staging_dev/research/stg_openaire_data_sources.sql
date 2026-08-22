{{ config(materialized='table') }}

select
    {{ dbt_utils.generate_surrogate_key(['openaire_id']) }} as openaire_data_sources_sk,
    ingestion_id,
    fetched_at,
    openaire_id,
    official_name,
    english_name,
    website_url,
    type as source_type,
    compatibility,
    subjects,
    issn_online,
    issn_printed,
    'openaire_data_sources' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Data_sources_bronze') }}
