{{ config(materialized='table') }}

-- Gold: full source dimension extended from staging_dev dim_source.
-- organisation_key, tier, data_level, reliability_score, data_usage_policy
-- are derived from known source metadata where possible.

select
    source_key,
    source_key                                     as source_natural_key,
    source_name,

    -- organisation_key resolved against dim_organisation when available
    cast(null as int64)                            as organisation_key,

    -- tier: 1 = global institutional, 2 = national, 3 = sub-national/community
    case
        when lower(source_name) like '%faostat%'
          or lower(source_name) like '%fews%'
          or lower(source_name) like '%wfp%'
          or lower(source_name) like '%openaire%'
          or lower(source_name) like '%worldbank%'
          or lower(source_name) like '%oecd%'       then 1
        when lower(source_name) like '%ilri%'
          or lower(source_name) like '%ifpri%'
          or lower(source_name) like '%cifor%'      then 2
        else 3
    end                                            as tier,

    case
        when lower(source_name) like '%faostat%'
          or lower(source_name) like '%wfp%'
          or lower(source_name) like '%openaire%'   then 'global'
        when lower(source_name) like '%fews%'       then 'national'
        else 'sub_national'
    end                                            as data_level,

    cast(null as int64)                            as reliability_score,
    cast(null as string)                           as data_usage_policy,
    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current

from {{ ref('dim_source') }}
