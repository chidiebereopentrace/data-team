{{ config(materialized='table') }}

with openaire as (
    select
        to_hex(md5(cast(organisation_id as string))) as organisation_key,
        organisation_id,
        legal_name,
        short_name,
        website_url,
        country_code,
        country_name,
        'openaire' as org_source,
        source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_openaire_organisations') }}
    where organisation_id is not null
    qualify row_number() over (
        partition by organisation_id
        order by fetched_at desc nulls last
    ) = 1
),

asti as (
    select
        to_hex(md5('asti|' || src.institution_norm)) as organisation_key,
        cast(null as string) as organisation_id,
        any_value(src.institution) as legal_name,
        cast(null as string) as short_name,
        cast(null as string) as website_url,
        cast(null as string) as country_code,
        cast(null as string) as country_name,
        'asti' as org_source,
        any_value(src.source_natural_key) as source_natural_key,
        current_timestamp() as loaded_at
    from (
        select
            institution,
            lower(trim(institution)) as institution_norm,
            source_natural_key
        from {{ ref('int_investment_asti_conformed') }}
        where institution is not null
    ) src
    group by src.institution_norm
)

select * from openaire
union all
select * from asti
