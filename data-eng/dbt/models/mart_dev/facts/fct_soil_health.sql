{{ config(materialized='table') }}

-- Gold: soil health fact from ISRIC soil data (staging_dev soil_dashboard_base).

select
    to_hex(md5(concat(
        coalesce(cast(lat_2 as string), ''), '|',
        coalesce(cast(lon_2 as string), '')
    )))                                            as soil_health_key,

    to_hex(md5(concat(
        coalesce(cast(lat_2 as string), ''), '|',
        coalesce(cast(lon_2 as string), ''), '|', ''
    )))                                            as geo_key,

    cast(null as int64)                            as time_key,
    to_hex(md5('isric_africa_soil_data'))          as source_key,
    to_hex(md5('0-5'))                             as soil_key,
    cast(null as string)                           as unit_key,
    cast(null as string)                           as audit_key,

    cast(null as numeric)                          as bdod,
    cast(null as numeric)                          as cec,
    cast(null as numeric)                          as clay_pct,
    cast(null as numeric)                          as sand_pct,
    cast(null as numeric)                          as silt_pct,
    cast(isda_avg_nitrogen as numeric)             as nitrogen,
    cast(isric_avg_ph_0_5cm as numeric)            as ph,
    cast(isric_avg_soc_0_5cm as numeric)           as soc,
    to_hex(md5(concat(
        coalesce(cast(lat_2 as string), ''), '|',
        coalesce(cast(lon_2 as string), '')
    )))                                            as source_record_id

from {{ ref('soil_dashboard_base') }}
