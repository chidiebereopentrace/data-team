{{ config(materialized='table') }}

-- ILRI Vegetation Survey + Feed Assessment Odisha.

with base as (
    select
        safe_cast(latitude as float64) as latitude,
        safe_cast(longitude as float64) as longitude,
        cast(survey_date as string) as survey_date,
        quantity_trees,
        quantity_shrubs,
        quantity_grass,
        leaves_trees,
        leaves_shrubs,
        leaves_grass,
        palatability_trees,
        palatability_shrubs,
        palatability_grass,
        carrying_capacity,
        currently_grazing,
        photo_id,
        cast(null as string) as household_id,
        cast(null as string) as agro_climatic_zone_id,
        'vegetation' as record_type,
        'ilri_vegetation_survey_v1' as source_natural_key
    from {{ source('raw_dev', 'ilri_vegetation_survey_v1') }}

    union all

    select
        GPS_North as latitude,
        GPS_East as longitude,
        cast(Date as string) as survey_date,
        cast(null as string) as quantity_trees,
        cast(null as string) as quantity_shrubs,
        cast(null as string) as quantity_grass,
        cast(null as string) as leaves_trees,
        cast(null as string) as leaves_shrubs,
        cast(null as string) as leaves_grass,
        cast(null as string) as palatability_trees,
        cast(null as string) as palatability_shrubs,
        cast(null as string) as palatability_grass,
        cast(null as string) as carrying_capacity,
        cast(null as string) as currently_grazing,
        cast(null as string) as photo_id,
        cast(HhId as string) as household_id,
        cast(Agro_Climatic_Zone_ID as string) as agro_climatic_zone_id,
        'feed_assessment' as record_type,
        'ilri_feed_assessment_odisha_database_2018_mel_public' as source_natural_key
    from {{ source('raw_dev', 'ilri_feed_assessment_odisha_database_2018_mel_public') }}
)

select
    to_hex(md5(to_json_string(base))) as ilri_vegetation_feed_sk,
    base.*,
    current_timestamp() as loaded_at
from base
