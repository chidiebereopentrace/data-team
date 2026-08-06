{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_adgo_database_mel_public') }}
