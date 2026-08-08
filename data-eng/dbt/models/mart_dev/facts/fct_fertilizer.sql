{{ config(materialized='incremental', unique_key='fertilizer_key') }}

-- Gold: fertilizer use fact from FAOSTAT fertilizers nutrient dataset.

select
    to_hex(md5(concat(
        coalesce(country_code, area, ''), '|',
        coalesce(item, ''), '|',
        coalesce(element, ''), '|',
        coalesce(cast(year as string), '')
    )))                                            as fertilizer_key,

    to_hex(md5(concat(
        coalesce(country_code, ''), '|', coalesce(area, ''), '|', ''
    )))                                            as geo_key,

    cast(safe_cast(year as int64) as int64)        as time_key,
    to_hex(md5('fao_fertilizers_nutrient_bronze')) as source_key,
    to_hex(md5(coalesce(item, '')))                as product_key,
    to_hex(md5(coalesce(unit, '')))                as unit_key,
    cast(value as numeric)                         as quantity,
    to_hex(md5(concat(
        coalesce(country_code, ''), '|', coalesce(item, ''), '|',
        coalesce(cast(year as string), '')
    )))                                            as source_record_id

from {{ source('raw_dev', 'fao_fertilizers_nutrient_bronze') }}
where value is not null

{% if is_incremental() %}
  and year > (select max(time_key) from {{ this }})
{% endif %}
