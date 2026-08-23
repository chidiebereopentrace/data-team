{{ config(materialized='table') }}

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
    country,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_ilri_food_hazards') }}
where foodborne_hazard is not null
