{{ config(materialized='table') }}

select
    org_id as organisation_id,
    legal_name,
    short_name,
    website_url,
    country_code,
    country_name,
    alternative_names,
    pids,
    fetched_at,
    'openaire_organisations' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Organizations_bronze') }}
