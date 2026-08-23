{{ config(materialized='table') }}

select
    dataset_id,
    ingested_timestamp,
    record_payload,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_un_peace_security') }}
