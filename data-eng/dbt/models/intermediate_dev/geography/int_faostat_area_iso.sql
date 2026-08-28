{{ config(materialized='table') }}

select * from {{ ref('int_faostat_area_reference') }}
