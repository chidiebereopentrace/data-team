{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'ilri_20141223nlaevaluationdataanonymouspublic') }}
