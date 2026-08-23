{{ config(materialized='table') }}

select
    organisation_id,
    legal_name,
    short_name,
    website_url,
    country_code,
    country_name,
    fetched_at,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_openaire_organisations') }}
where organisation_id is not null
