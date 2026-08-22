{{ config(materialized='table') }}

-- ILRI Vendor and Consumer surveys (Burkina Faso + Ethiopia).

with base as (
    select
        id_outlet_t0 as outlet_id,
        today_t0 as survey_date,
        age_t0 as respondent_age,
        sex_t0 as respondent_sex,
        cast(null as string) as consumer_id,
        'vendor' as respondent_type,
        'Burkina Faso' as country,
        'ilri_bf_rct_vendorsurvey_t0t1_anonymous_final' as source_natural_key
    from {{ source('raw_dev', 'ilri_bf_rct_vendorsurvey_t0t1_anonymous_final') }}

    union all

    select
        cast(ID_outlet_t0 as string) as outlet_id,
        today_t0 as survey_date,
        cast(null as string) as respondent_age,
        cast(null as string) as respondent_sex,
        NEW_ID_t0 as consumer_id,
        'consumer' as respondent_type,
        'Burkina Faso' as country,
        'ilri_burkinafaso_consumersurvey_t0t1data_wide_anonymous' as source_natural_key
    from {{ source('raw_dev', 'ilri_burkinafaso_consumersurvey_t0t1data_wide_anonymous') }}

    union all

    select
        cast(ID_outlet_t0 as string) as outlet_id,
        today_t0 as survey_date,
        cast(null as string) as respondent_age,
        cast(null as string) as respondent_sex,
        NEW_ID as consumer_id,
        'consumer' as respondent_type,
        'Ethiopia' as country,
        'ilri_ethiopia_consumersurvey_t0t1data_wide_anonymous' as source_natural_key
    from {{ source('raw_dev', 'ilri_ethiopia_consumersurvey_t0t1data_wide_anonymous') }}
)

select
    to_hex(md5(to_json_string(base))) as ilri_vendor_consumer_sk,
    base.*,
    current_timestamp() as loaded_at
from base
