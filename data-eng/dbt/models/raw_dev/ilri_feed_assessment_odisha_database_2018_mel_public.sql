{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_feed_assessment_odisha_database_2018_mel_public') }}
