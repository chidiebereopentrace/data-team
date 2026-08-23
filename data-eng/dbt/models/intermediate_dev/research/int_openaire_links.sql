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
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_openaire_product_links') }}
