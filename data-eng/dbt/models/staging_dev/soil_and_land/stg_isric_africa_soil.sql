{{ config(materialized='table') }}

select
    latitude,
    longitude,
    cast(fetched_at as date) as fetched_date,
    bdod_0_5cm,
    bdod_5_15cm,
    bdod_15_30cm,
    bdod_30_60cm,
    bdod_60_100cm,
    cec_0_5cm,
    cec_5_15cm,
    cec_15_30cm,
    cec_30_60cm,
    cec_60_100cm,
    clay_0_5cm,
    clay_5_15cm,
    clay_15_30cm,
    clay_30_60cm,
    clay_60_100cm,
    nitrogen_0_5cm,
    nitrogen_5_15cm,
    nitrogen_15_30cm,
    nitrogen_30_60cm,
    nitrogen_60_100cm,
    phh2o_0_5cm,
    phh2o_5_15cm,
    phh2o_15_30cm,
    phh2o_30_60cm,
    phh2o_60_100cm,
    sand_0_5cm,
    sand_5_15cm,
    sand_15_30cm,
    sand_30_60cm,
    sand_60_100cm,
    silt_0_5cm,
    silt_5_15cm,
    silt_15_30cm,
    silt_30_60cm,
    silt_60_100cm,
    soc_0_5cm,
    soc_5_15cm,
    soc_15_30cm,
    soc_30_60cm,
    soc_60_100cm,
    'ISRIC_Africa_Soil' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'isric_africa_soil_data') }}
where not (
    bdod_0_5cm is null and bdod_5_15cm is null and bdod_15_30cm is null and bdod_30_60cm is null and bdod_60_100cm is null
    and cec_0_5cm is null and cec_5_15cm is null and cec_15_30cm is null and cec_30_60cm is null and cec_60_100cm is null
    and clay_0_5cm is null and clay_5_15cm is null and clay_15_30cm is null and clay_30_60cm is null and clay_60_100cm is null
    and nitrogen_0_5cm is null and nitrogen_5_15cm is null and nitrogen_15_30cm is null and nitrogen_30_60cm is null and nitrogen_60_100cm is null
    and phh2o_0_5cm is null and phh2o_5_15cm is null and phh2o_15_30cm is null and phh2o_30_60cm is null and phh2o_60_100cm is null
    and sand_0_5cm is null and sand_5_15cm is null and sand_15_30cm is null and sand_30_60cm is null and sand_60_100cm is null
    and silt_0_5cm is null and silt_5_15cm is null and silt_15_30cm is null and silt_30_60cm is null and silt_60_100cm is null
    and soc_0_5cm is null and soc_5_15cm is null and soc_15_30cm is null and soc_30_60cm is null and soc_60_100cm is null
)
