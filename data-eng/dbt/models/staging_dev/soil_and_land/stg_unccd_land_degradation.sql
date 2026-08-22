{{ config(materialized='table') }}

with base as (
    select
        goal,
        target,
        indicator,
        series,
        seriesDescription as series_description,
        seriesCount as series_count,
        geoAreaCode as geo_area_code,
        geoAreaName as geo_area_name,
        timePeriodStart as time_period_start,
        value,
        valueType as value_type,
        array_to_string(time_detail, ',') as time_detail,
        timeCoverage as time_coverage,
        upperBound as upper_bound,
        lowerBound as lower_bound,
        'UNCCD_land_degradation' as source_natural_key,
        current_timestamp() as loaded_at
    from {{ source('raw_dev', 'unccd_land_degradation_bulk') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['geo_area_code', 'indicator', 'series', 'time_period_start', 'time_detail']) }} as unccd_land_degradation_sk,
    base.*
from base
