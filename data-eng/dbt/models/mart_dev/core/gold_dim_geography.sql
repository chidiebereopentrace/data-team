{{ config(materialized='table') }}

-- Gold: full geography dimension extended from staging_dev dim_geography.
-- FEWS classifications supplies national-level metadata (fnid, fewsnet_region, geographic_unit).
-- FEWS prices supplies sub-national admin hierarchy and market names.

with base as (
    select
        geography_key,
        country_code,
        country_name,
        admin_region
    from {{ ref('dim_geography') }}
),

-- FEWS classifications: national-level only (no admin_1/admin_2 columns)
fews_national as (
    select distinct
        country_code,
        fnid,
        geographic_unit,
        geographic_unit_name,
        geographic_group,
        fewsnet_region
    from {{ source('landing', 'FEWS_NET_food_security_classifications_time_series_data') }}
    where country_code is not null
),

-- FEWS prices: has admin_1, admin_2, and market names
fews_prices as (
    select distinct
        country_code,
        admin_1,
        admin_2,
        market                                     as market_name
    from {{ source('landing', 'FEWS_NET_market_Prices_time_series_data') }}
    where country_code is not null
)

select
    b.geography_key,

    to_hex(md5(concat(
        coalesce(b.country_code, ''), '|',
        coalesce(b.admin_region, '')
    )))                                            as geo_natural_key,

    b.country_code,
    b.country_name,
    b.country_name                                 as admin_0,
    b.admin_region                                 as admin_1,
    fp.admin_2,
    cast(null as string)                           as admin_3,
    cast(null as string)                           as admin_4,
    fn.fnid,
    fn.geographic_unit,
    fn.geographic_unit_name,
    fn.geographic_group,
    fn.fewsnet_region,
    cast(null as float64)                          as latitude,
    cast(null as float64)                          as longitude,
    cast(null as float64)                          as elevation_meters,
    fp.market_name,
    cast(null as string)                           as border_point,

    -- AEZ enrichment (populated by bridge_geography_aez; left null here)
    cast(null as string)                           as aez_code,
    cast(null as string)                           as aez_name,
    cast(null as string)                           as aez_version,
    cast(null as string)                           as aez_source,

    date('2000-01-01')                             as valid_from,
    cast(null as date)                             as valid_to,
    true                                           as is_current,
    current_timestamp()                            as created_at,
    current_timestamp()                            as updated_at

from base b
left join fews_national fn
    on b.country_code = fn.country_code
left join fews_prices fp
    on b.country_code = fp.country_code
    and b.admin_region = fp.admin_1
