{{ config(materialized='table') }}

-- ILRI Foodborne Hazards Meta-Analysis (Burkina Faso).
-- Ethiopia systematic review excluded (mostly unnamed columns).

with base as (
    select
        author_year,
        publication_year,
        study_site,
        sampling_points,
        samples_type,
        sample_subgroup,
        foodborne_hazard,
        total_samples,
        positive_samples,
        mean_cfu_per_g_log,
        standard_deviations,
        'Burkina Faso' as country,
        'ilri_dataset_burkina_faso_food_hazards_review_meta_analysis' as source_natural_key
    from {{ source('raw_dev', 'ilri_dataset_burkina_faso_food_hazards_review_meta_analysis') }}
)

select
    to_hex(md5(to_json_string(base))) as ilri_food_hazards_sk,
    base.*,
    current_timestamp() as loaded_at
from base
