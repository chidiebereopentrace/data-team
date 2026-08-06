{{ config(materialized='table') }}

select
    openaire_id,
    title,
    entity_type,
    pub_date,
    publisher,
    language,
    rel_name,
    target_id,
    target_type,
    fetched_at,
    'openaire_product_links' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'openaire_agriculture_and_environment_Research_publications_Product_links_bronze') }}
