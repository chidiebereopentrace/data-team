{{ config(materialized='table') }}

select
    PLOT as plot_id,
    Treatment as treatment,
    Soiltype as soil_type,
    `%C` as carbon_pct,
    `%N` as nitrogen_pct,
    `%P` as phosphorus_pct,
    `%K` as potassium_pct,
    `%Ca` as calcium_pct,
    `%Mg` as magnesium_pct,
    _source_doi as source_doi,
    'cifor_icraf_raw' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'cifor_icraf_raw') }}
