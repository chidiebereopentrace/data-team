{{ config(materialized='table') }}

-- Gold: links households to diseases reported in their livestock.
-- Populated once household-level staging data is available from ILRI surveys.
-- Returns an empty result set with the correct schema until upstream is wired.

select
    cast(null as string)                           as household_key,
    cast(null as string)                           as disease_key

from (select 1 as _dummy) where false
