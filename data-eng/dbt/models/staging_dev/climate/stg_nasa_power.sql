{{ config(materialized='table') }}

-- NASA POWER daily summary + hourly; shared radiation names with hourly mapped in.

with base as (
    select
        country_code,
        country_name,
        admin_region,
        cast(null as float64) as latitude,
        cast(null as float64) as longitude,
        elevation_meters,
        par_solar_at_noon,
        shortwave_irradiance_at_noon,
        uva_radiation_at_noon,
        uvb_radiation_at_noon,
        cast(null as timestamp) as observation_time,
        fetched_at,
        processed_at,
        'NASA_POWER_daily' as source_natural_key
    from {{ source('raw_dev', 'africa_nasa_power_daily_summary_bronze') }}

    union all

    select
        country_code,
        cast(null as string) as country_name,
        admin_region,
        latitude,
        longitude,
        elevation as elevation_meters,
        par_total as par_solar_at_noon,
        shortwave_irradiance as shortwave_irradiance_at_noon,
        uva_irradiance as uva_radiation_at_noon,
        uvb_irradiance as uvb_radiation_at_noon,
        observation_time,
        fetched_at,
        cast(null as timestamp) as processed_at,
        'NASA_POWER_hourly' as source_natural_key
    from {{ source('raw_dev', 'africa_nasa_power_hourly_bronze') }}
)

select
    to_hex(md5(to_json_string(base))) as nasa_power_sk,
    base.*,
    current_timestamp() as loaded_at
from base
