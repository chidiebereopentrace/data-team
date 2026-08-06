{{ config(materialized='table') }}

-- ILRI other / lower-priority surveys (catch-all provenance contract).

select
    cast(interviewername as string) as enumerator,
    start_time_user as survey_start,
    cast(deviceid as string) as deviceid,
    village_filter as village,
    block_filter as block,
    cast(null as string) as household_id,
    cast(null as string) as country,
    'adgo' as survey_type,
    'ilri_adgo_database_mel_public' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'ilri_adgo_database_mel_public') }}

union all

select
    cast(null as string) as enumerator,
    cast(null as string) as survey_start,
    cast(null as string) as deviceid,
    cast(null as string) as village,
    cast(null as string) as block,
    cast(null as string) as household_id,
    cast(null as string) as country,
    'agripreneur' as survey_type,
    'ilri_agripreneurbaselinedata_public' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'ilri_agripreneurbaselinedata_public') }}

union all

select
    cast(null as string) as enumerator,
    cast(null as string) as survey_start,
    cast(null as string) as deviceid,
    cast(null as string) as village,
    cast(null as string) as block,
    cast(null as string) as household_id,
    cast(null as string) as country,
    'dairy_cashflow' as survey_type,
    'ilri_dairy_enterprise_cash_flows_ifamr_paperanon' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'ilri_dairy_enterprise_cash_flows_ifamr_paperanon') }}

union all

select
    cast(null as string) as enumerator,
    date_s as survey_start,
    cast(null as string) as deviceid,
    vil as village,
    cast(null as string) as block,
    qid as household_id,
    'Tanzania' as country,
    'tanzania_dairy_forum' as survey_type,
    'ilri_databasetanzaniadairydevelopmentforumevaluationanonymous150119_public' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'ilri_databasetanzaniadairydevelopmentforumevaluationanonymous150119_public') }}
