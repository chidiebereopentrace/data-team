{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_asfv_genotypeix_plasmid_library') }}
