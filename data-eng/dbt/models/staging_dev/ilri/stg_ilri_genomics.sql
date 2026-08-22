{{ config(materialized='table') }}

with base as (
    select
        cast(`Unnamed: 0` as string) as col_0,
        cast(`Unnamed: 1` as string) as col_1,
        cast(`Unnamed: 2` as string) as col_2,
        cast(`Unnamed: 3` as string) as col_3,
        'accessions' as genomics_domain,
        'ilri_accessions' as source_natural_key
    from {{ source('raw_dev', 'ilri_accessions') }}

    union all

    select
        cast(ID as string) as col_0,
        cast(ANIMAL as string) as col_1,
        cast(MYD as string) as col_2,
        cast(null as string) as col_3,
        'aliloo_phenotypes' as genomics_domain,
        'ilri_aliloo_phenotypes_public' as source_natural_key
    from {{ source('raw_dev', 'ilri_aliloo_phenotypes_public') }}
)

select
    to_hex(md5(to_json_string(base))) as ilri_genomics_sk,
    base.*,
    current_timestamp() as loaded_at
from base
