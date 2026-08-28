{{ config(materialized='table') }}

with base as (
    select
        to_hex(md5(
            coalesce(l.openaire_id, '') || '|' ||
            coalesce(l.target_id, '') || '|' ||
            coalesce(l.rel_name, '')
        )) as research_person_bridge_key,
        l.openaire_id,
        p.person_key,
        l.target_id,
        l.rel_name,
        l.target_type,
        l.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_openaire_links') }} l
    left join {{ ref('dim_person') }} p
        on p.person_id = l.target_id
    where l.target_id is not null
      and (
           lower(coalesce(l.target_type, '')) like '%person%'
        or lower(coalesce(l.rel_name, '')) like '%author%'
      )
)

select *
from base
qualify row_number() over (
    partition by research_person_bridge_key
    order by person_key nulls last, rel_name
) = 1
