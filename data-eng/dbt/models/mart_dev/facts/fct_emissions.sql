{{ config(materialized='table') }}

-- Gold: emissions fact from ClimateWatch emission pathways (landing source).

select
    to_hex(md5(concat(
        coalesce(iso_code2, location, ''), '|',
        coalesce(category, ''), '|',
        coalesce(subcategory, ''), '|',
        coalesce(indicator, ''), '|',
        coalesce(model, '')
    )))                                            as emission_key,

    to_hex(md5(concat(
        coalesce(iso_code2, ''), '|', coalesce(location, ''), '|', ''
    )))                                            as geo_key,

    cast(null as int64)                            as time_key,

    to_hex(md5('climatewatch_emission_pathways_raw'))
                                                   as source_key,

    to_hex(md5(coalesce(category, '')))            as emission_category_key,

    to_hex(md5(coalesce(unit, '')))                as unit_key,
    cast(null as string)                           as audit_key,

    cast(null as numeric)                          as value,
    to_hex(md5(concat(
        coalesce(iso_code2, ''), '|', coalesce(category, ''), '|', coalesce(indicator, '')
    )))                                            as source_record_id

from {{ source('landing', 'climatewatch_emission_pathways_raw') }}
where location is not null

{% if is_incremental() %}
  and model is not null
{% endif %}
