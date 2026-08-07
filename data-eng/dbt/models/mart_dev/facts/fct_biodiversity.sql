{{ config(materialized='incremental', unique_key='biodiversity_key') }}

-- Gold: biodiversity occurrence fact from GBIF Africa occurrence dataset.

select
    to_hex(md5(concat(
        coalesce(cast(gbif_id as string), ''), '|',
        coalesce(species, ''), '|',
        coalesce(countrycode, '')
    )))                                            as biodiversity_key,

    to_hex(md5(concat(
        coalesce(countrycode, ''), '|',
        coalesce(stateprovince, ''), '|', ''
    )))                                            as geo_key,

    cast(year as int64)                            as time_key,
    to_hex(md5('gbif_biodiversity_occurrence'))    as source_key,
    cast(individual_count as numeric)              as occurrence_count,
    cast(null as numeric)                          as rarity_score,
    cast(gbif_id as string)                        as source_record_id

from {{ source('raw_dev', 'gbif_biodiversity_occurrence') }}
where gbif_id is not null

{% if is_incremental() %}
  and year > (select max(time_key) from {{ this }})
{% endif %}
