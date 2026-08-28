{{ config(materialized='table') }}

select
    to_hex(md5(source_natural_key)) as source_key,
    source_natural_key,
    organisation_name,
    tier,
    default_data_level,
    producer_scale,
    current_timestamp() as loaded_at
from {{ ref('int_source_registry') }}
