{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(src.source_natural_key, '') || '|' || coalesce(src.household_id, '')
    )) as household_key,
    src.household_id,
    src.country,
    src.region,
    src.district,
    src.village_code,
    src.household_type,
    src.head_education_level,
    src.source_natural_key,
    current_timestamp() as loaded_at
from (
    select
        household_id,
        country,
        region,
        district,
        village_code,
        household_type,
        head_education_level,
        source_natural_key
    from {{ ref('int_ilri_household_conformed') }}
    where household_id is not null

    union all

    select
        household_id,
        country,
        cast(null as string) as region,
        cast(null as string) as district,
        cast(null as string) as village_code,
        cast(null as string) as household_type,
        cast(null as string) as head_education_level,
        source_natural_key
    from {{ ref('int_i4i_insurance_conformed') }}
    where household_id is not null
      and record_type = 'farmer'
) src
qualify row_number() over (
    partition by src.source_natural_key, src.household_id
    order by src.household_id
) = 1
