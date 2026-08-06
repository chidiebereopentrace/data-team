{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_dairy_enterprise_cash_flows_ifamr_paperanon') }}
