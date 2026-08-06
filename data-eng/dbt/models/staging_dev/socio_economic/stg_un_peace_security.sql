{{ config(
    materialized='table',
    enabled=false
) }}

-- Payload-style records; low analytical value. Kept disabled for completeness.

select
    dataset_id,
    ingested_timestamp,
    record_payload,
    'un_peace_security_records' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'un_peace_security_records') }}
