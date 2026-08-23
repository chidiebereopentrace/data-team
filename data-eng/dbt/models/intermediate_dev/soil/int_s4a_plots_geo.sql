{{ config(materialized='table') }}

select
    plot_code,
    plot_id,
    country_code,
    survey_date,
    safe_cast(altitude as float64) as altitude,
    safe_cast(longitude as float64) as longitude,
    safe_cast(latitude as float64) as latitude,
    tsu_id,
    obstruction_layer,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_s4a_field_surveys') }}
where safe_cast(longitude as float64) between -25 and 60
  and safe_cast(latitude as float64) between -35 and 38
