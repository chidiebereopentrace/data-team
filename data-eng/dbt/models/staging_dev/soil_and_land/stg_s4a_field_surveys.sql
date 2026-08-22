{{ config(materialized='table') }}

-- S4A field surveys: soil surface theme (other themes can be unioned later).

select
    {{ dbt_utils.generate_surrogate_key(['plot_code', 'survey_date', 'tsu_id']) }} as s4a_field_surveys_sk,
    plot_code,
    plot_id,
    country_code,
    survey_date,
    altitude,
    longitude,
    latitude,
    tsu_id,
    obstruct_lyr as obstruction_layer,
    's4a_field_surveys' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 's4a_field_soil_surface_info') }}
