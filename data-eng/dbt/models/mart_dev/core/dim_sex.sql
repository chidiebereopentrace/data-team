{{ config(materialized='table') }}

select
    sex_key,
    sex_label,
    current_timestamp() as loaded_at
from unnest([
    struct('male' as sex_key, 'Male' as sex_label),
    struct('female' as sex_key, 'Female' as sex_label),
    struct('total' as sex_key, 'Total' as sex_label),
    struct('unknown' as sex_key, 'Unknown' as sex_label)
])
