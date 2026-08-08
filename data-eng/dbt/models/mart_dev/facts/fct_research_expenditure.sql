{{ config(materialized='table') }}

-- Gold: research expenditure from OECD food/ag R&D data.
-- OECD_Food_data_Africa_NEW not yet ingested into BigQuery; stub until ingestion is complete.

select
    cast(null as string)                           as research_exp_key,
    cast(null as string)                           as geo_key,
    cast(null as int64)                            as time_key,
    cast(null as string)                           as source_key,
    cast(null as string)                           as organisation_key,
    cast(null as string)                           as unit_key,
    cast(null as numeric)                          as expenditure_amount,
    cast(null as string)                           as source_record_id

from (select 1 as _dummy) where false
