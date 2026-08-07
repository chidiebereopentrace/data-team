{{ config(materialized='table') }}

-- Gold: population dynamics fact from South Africa ward demographics (ArcGIS).
-- Covers ward-level population counts; national-level data added as sources mature.

select
    to_hex(md5(concat(
        coalesce(wardid, ''), '|', coalesce(cast(year as string), '')
    )))                                            as population_fact_key,

    to_hex(md5(concat(
        coalesce(provinceco, ''), '|',
        coalesce(provincena, ''), '|',
        coalesce(localmunic, '')
    )))                                            as geo_key,

    cast(year as int64)                            as time_key,
    to_hex(md5('arcgis_south_africa_wards_demographics_2ce07'))
                                                   as source_key,
    cast(total as numeric)                         as population_count,
    coalesce(wardid, cast(wardinn as string))      as source_record_id

from {{ source('raw_dev', 'arcgis_south_africa_wards_demographics_2ce07') }}
where total is not null
