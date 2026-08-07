{{ config(materialized='table') }}

-- Gold: audit dimension capturing pipeline run metadata.
-- One record per dbt run; downstream facts reference audit_key for lineage.

select
    to_hex(md5(concat(
        coalesce(cast(current_timestamp() as string), ''),
        '|open_trace'
    )))                                            as audit_key,
    current_timestamp()                            as ingestion_timestamp,
    'open_trace_dbt'                               as source_system,
    cast(null as string)                           as source_file_name,
    cast(null as string)                           as batch_id,
    cast(null as string)                           as pipeline_run_id,
    current_timestamp()                            as processed_at,
    cast(null as string)                           as record_hash,
    false                                          as is_deleted
