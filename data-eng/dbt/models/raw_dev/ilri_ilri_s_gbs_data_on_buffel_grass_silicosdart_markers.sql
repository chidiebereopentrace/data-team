{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_ilri_s_gbs_data_on_buffel_grass_silicosdart_markers') }}
