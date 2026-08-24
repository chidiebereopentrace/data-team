{{ config(materialized='table') }}

select
    gbif_id,
    scientific_name,
    kingdom,
    family,
    genus,
    species,
    country_code,
    locality,
    state_province,
    individual_count,
    latitude,
    longitude,
    year,
    month,
    day,
    event_date,
    basis_of_record,
    rich_all,
    rar_all,
    area_protected,
    geometry_wkt,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_biodiversity') }}
where (
        country_code is not null
        or (
            longitude between -25 and 60
            and latitude between -35 and 38
        )
      )
