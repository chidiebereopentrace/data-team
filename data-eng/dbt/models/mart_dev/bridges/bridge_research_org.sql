{{ config(materialized='table') }}

with base as (
    select
        to_hex(md5(
            coalesce(l.openaire_id, '') || '|' ||
            coalesce(l.target_id, '') || '|' ||
            coalesce(l.rel_name, '')
        )) as research_org_bridge_key,
        l.openaire_id,
        o.organisation_key,
        l.target_id,
        l.rel_name,
        l.target_type,
        l.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_openaire_links') }} l
    left join {{ ref('dim_organisation') }} o
        on o.organisation_id = l.target_id
       and o.org_source = 'openaire'
    where l.target_id is not null
      and (
           lower(coalesce(l.target_type, '')) like '%org%'
        or lower(coalesce(l.rel_name, '')) like '%org%'
      )
)

select *
from base
qualify row_number() over (
    partition by research_org_bridge_key
    order by organisation_key nulls last, rel_name
) = 1
