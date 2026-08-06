{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_dataset_burkina_faso_food_hazards_review_meta_analysis') }}
