{{ config(materialized='table') }}

select
    {{ dbt_utils.generate_surrogate_key(['dataset_id']) }} as un_peace_security_sk,
    dataset_id,
    ingested_timestamp,
    record_payload,
    'un_peace_security_records' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'un_peace_security_records') }}
