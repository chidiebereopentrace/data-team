{{ config(materialized='table') }}

select
    geo_key,
    aez_code,
    aez_name,
    aez_version,
    aez_source,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_aez') }}
where geo_key is not null
  and aez_code is not null
