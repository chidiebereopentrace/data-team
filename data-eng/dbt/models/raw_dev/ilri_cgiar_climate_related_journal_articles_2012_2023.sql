{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_cgiar_climate_related_journal_articles_2012_2023') }}
