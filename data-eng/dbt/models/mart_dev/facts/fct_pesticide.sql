{{ config(materialized='incremental', unique_key='pesticide_key') }}

-- Gold: pesticide use fact from FAOSTAT pesticides use dataset.

select
    to_hex(md5(concat(
        coalesce(country_code, area, ''), '|',
        coalesce(item, ''), '|',
        coalesce(element, ''), '|',
        coalesce(cast(year as string), '')
    )))                                            as pesticide_key,

    to_hex(md5(concat(
        coalesce(country_code, ''), '|', coalesce(area, ''), '|', ''
    )))                                            as geo_key,

    cast(safe_cast(year as int64) as int64)        as time_key,
    to_hex(md5('fao_pesticides_use_bronze'))       as source_key,
    to_hex(md5(coalesce(item, '')))                as product_key,
    cast(null as string)                           as pest_key,
    to_hex(md5(coalesce(unit, '')))                as unit_key,
    cast(value as numeric)                         as quantity,
    cast(null as numeric)                          as application_frequency,
    cast(null as numeric)                          as eiq_score,
    cast(null as numeric)                          as eiq_field_use,
    cast(null as bool)                             as treatment_misuse_flag,
    to_hex(md5(concat(
        coalesce(country_code, ''), '|', coalesce(item, ''), '|',
        coalesce(cast(year as string), '')
    )))                                            as source_record_id

from {{ source('raw_dev', 'fao_pesticides_use_bronze') }}
where value is not null

{% if is_incremental() %}
  and year > (select max(time_key) from {{ this }})
{% endif %}
