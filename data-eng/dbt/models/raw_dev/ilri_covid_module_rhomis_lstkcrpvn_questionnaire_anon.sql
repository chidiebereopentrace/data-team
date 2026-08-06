{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_covid_module_rhomis_lstkcrpvn_questionnaire_anon') }}
