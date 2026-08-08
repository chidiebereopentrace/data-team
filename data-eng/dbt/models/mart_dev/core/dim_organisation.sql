{{ config(materialized='table') }}

-- Gold: organisation dimension sourced from OpenAire organisations (raw_dev).
-- Covers research institutions, donors, and implementing partners.

select
    to_hex(md5(coalesce(org_id, legal_name, ''))) as organisation_key,
    coalesce(org_id, '')                           as organisation_natural_key,
    legal_name,
    short_name,
    cast(null as string)                           as organisation_type,
    country_code,
    website_url,
    false                                          as is_donor,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Organizations_bronze') }}
where coalesce(org_id, legal_name, short_name) is not null
