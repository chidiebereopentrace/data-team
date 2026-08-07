{{ config(materialized='table') }}

-- Gold: technology fact extended from staging_dev fact_technology.

select
    fact_technology_key                            as technology_fact_key,
    geography_key                                  as geo_key,
    period_key                                     as time_key,
    source_key,
    cast(null as string)                           as organisation_key,
    technology_key,
    unit_key,
    cast(null as string)                           as audit_key,
    cast(null as numeric)                          as machinery_stock,
    cast(null as numeric)                          as rd_expenditure,
    technology_value                               as rd_intensity,
    fact_technology_key                            as source_record_id

from {{ ref('fact_technology') }}
