{{ config(materialized='table') }}

-- Gold: Human Development Index fact from UNDP/Africa HDI dataset.

select
    to_hex(md5(concat(
        coalesce(alpha_3_code, country, ''), '|', coalesce(year, '')
    )))                                            as hdi_key,

    to_hex(md5(concat(
        coalesce(alpha_3_code, ''), '|', coalesce(country, ''), '|', ''
    )))                                            as geo_key,

    cast(safe_cast(year as int64) as int64)        as time_key,
    to_hex(md5('africa_Human_development_index'))  as source_key,
    cast(null as string)                           as audit_key,
    cast(safe_cast(index as float64) as numeric)   as hdi_value,
    to_hex(md5(concat(
        coalesce(alpha_3_code, ''), '|', coalesce(year, '')
    )))                                            as source_record_id

from {{ source('raw_dev', 'africa_Human_development_index') }}
where index is not null
