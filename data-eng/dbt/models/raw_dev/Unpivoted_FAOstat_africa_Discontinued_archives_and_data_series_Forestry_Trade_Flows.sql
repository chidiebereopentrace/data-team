{{ config(materialized='view', enabled=false) }}

select
    *
from {{ source('raw_dev', 'Unpivoted_FAOstat_africa_Discontinued_archives_and_data_series_Forestry_Trade_Flows') }}
