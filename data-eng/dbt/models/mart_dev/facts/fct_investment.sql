{{ config(materialized='table') }}

-- Gold: investment fact extended from staging_dev fact_enterprise_investment.

select
    fact_enterprise_investment_key                 as investment_key,
    geography_key                                  as geo_key,
    period_key                                     as time_key,
    source_key,
    cast(null as string)                           as organisation_key,
    unit_key,
    cast(null as string)                           as audit_key,
    enterprise_investment_value                    as amount,
    fact_enterprise_investment_key                 as source_record_id

from {{ ref('fact_enterprise_investment') }}
