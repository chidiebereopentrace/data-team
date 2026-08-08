{{ config(materialized='table') }}

-- Gold: germplasm accession fact from crop germplasm Africa dataset.

select
    to_hex(md5(concat(
        coalesce(cast(id as string), ''), '|',
        coalesce(taxon, ''), '|',
        coalesce(st_astext(geography), '')
    )))                                            as germplasm_key,

    to_hex(md5(concat(
        coalesce(st_astext(geography), ''), '|', '', '|', ''
    )))                                            as geo_key,

    cast(null as int64)                            as time_key,
    to_hex(md5('crop_germplasm_africa'))           as source_key,
    to_hex(md5(coalesce(taxon, '')))               as product_key,
    cast(null as string)                           as breed_variety_key,
    cast(1 as int64)                               as accession_count,
    cast(id as string)                             as source_record_id

from {{ source('raw_dev', 'crop_germplasm_africa') }}
where id is not null
