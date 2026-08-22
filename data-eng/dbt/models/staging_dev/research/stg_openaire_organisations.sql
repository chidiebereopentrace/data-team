{{ config(materialized='table') }}

with base as (
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
)

select
    {{ dbt_utils.generate_surrogate_key(['organisation_id']) }} as openaire_organisations_sk,
    base.*
from base
