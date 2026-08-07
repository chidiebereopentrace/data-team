{{ config(materialized='incremental', unique_key='climate_key') }}

-- Gold: climate fact extended from staging_dev fact_climate.

select
    fact_climate_key                               as climate_key,
    geography_key                                  as geo_key,
    period_key                                     as time_key,
    source_key,
    cast(null as string)                           as sensor_key,
    cast(null as string)                           as audit_key,
    climate_value                                  as temperature,
    cast(null as numeric)                          as precipitation,
    cast(null as numeric)                          as humidity,
    cast(null as numeric)                          as radiation,
    cast(null as numeric)                          as par_solar,
    cast(null as numeric)                          as shortwave_irradiance,
    cast(null as numeric)                          as uva_radiation,
    cast(null as numeric)                          as uvb_radiation,
    fact_climate_key                               as source_record_id

from {{ ref('fact_climate') }}

{% if is_incremental() %}
  where period_key > (select max(time_key) from {{ this }})
{% endif %}
