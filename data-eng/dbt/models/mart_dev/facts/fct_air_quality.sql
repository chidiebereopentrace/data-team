{{ config(materialized='incremental', unique_key='air_quality_key') }}

-- Gold: air quality fact from Nakuru air quality sensor archive.

select
    to_hex(md5(concat(
        coalesce(cast(sensor_id as string), ''), '|',
        coalesce(timestamp, '')
    )))                                            as air_quality_key,

    to_hex(md5(concat(
        coalesce(cast(lat as string), ''), '|',
        coalesce(cast(lon as string), ''), '|', ''
    )))                                            as geo_key,

    cast(null as int64)                            as time_key,
    to_hex(md5('nakuru_air_quality_archive'))      as source_key,
    to_hex(md5(cast(sensor_id as string)))         as sensor_key,
    cast(null as string)                           as audit_key,
    cast(pm10 as numeric)                          as pm10,
    cast(pm2_5 as numeric)                         as pm2_5,
    cast(humidity_pct as numeric)                  as humidity_pct,
    cast(temp_c as numeric)                        as temp_c,
    to_hex(md5(concat(
        coalesce(cast(sensor_id as string), ''), '|', coalesce(timestamp, '')
    )))                                            as source_record_id

from {{ source('raw_dev', 'nakuru_air_quality_archive') }}
where pm10 is not null or pm2_5 is not null

{% if is_incremental() %}
  and timestamp > (select cast(max(time_key) as string) from {{ this }})
{% endif %}
