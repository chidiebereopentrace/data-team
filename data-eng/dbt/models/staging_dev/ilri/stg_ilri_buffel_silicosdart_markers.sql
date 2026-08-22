{{ config(materialized='table') }}

with base as (
    select
        *,
        'ilri_ilri_s_gbs_data_on_buffel_grass_silicosdart_markers' as source_natural_key
    from {{ source('raw_dev', 'ilri_ilri_s_gbs_data_on_buffel_grass_silicosdart_markers') }}
)

select
    to_hex(md5(to_json_string(base))) as ilri_buffel_silicosdart_markers_sk,
    base.*,
    current_timestamp() as loaded_at
from base
