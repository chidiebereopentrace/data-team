{{ config(materialized='table') }}

select
    country,
    country_code,
    admin_1,
    admin_2,
    market_name,
    product_name,
    cpcv2,
    price_type,
    unit,
    currency,
    value as price_value,
    common_unit_price,
    common_currency_price,
    year,
    month,
    cast(null as string) as faostat_months_label,
    cast(null as string) as faostat_months_code,
    cast(null as int64) as area_code,
    cast(null as int64) as area_code_m49,
    cast(null as string) as country_id,
    cast(null as string) as admin1_id,
    'fews' as price_source,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_fews_market_prices') }}
where value is not null

union all

select
    w.country,
    cast(null as string) as country_code,
    w.admin1_name as admin_1,
    cast(null as string) as admin_2,
    w.market_name,
    w.product_name,
    cast(null as string) as cpcv2,
    w.price_type,
    w.unit,
    w.currency,
    w.value as price_value,
    cast(null as float64) as common_unit_price,
    cast(null as float64) as common_currency_price,
    w.year,
    w.month,
    cast(null as string) as faostat_months_label,
    cast(null as string) as faostat_months_code,
    cast(null as int64) as area_code,
    cast(null as int64) as area_code_m49,
    cast(w.country_id as string) as country_id,
    cast(w.admin1_id as string) as admin1_id,
    'wfp' as price_source,
    w.source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_wfp_vampire_prices') }} w
left join {{ ref('int_ref_country') }} rc_name
    on lower(trim(rc_name.country_name)) = lower(trim(w.country))
left join {{ ref('int_ref_country') }} rc_alias
    on rc_alias.country_iso3 = {{ geo_faostat_country_name_iso3('w.country') }}
where w.value is not null
  and coalesce(rc_name.in_africa_scope, rc_alias.in_africa_scope, false)

union all

select
    country_name as country,
    cast(null as string) as country_code,
    cast(null as string) as admin_1,
    cast(null as string) as admin_2,
    cast(null as string) as market_name,
    product_name,
    item_code_cpc as cpcv2,
    element as price_type,
    unit,
    currency,
    safe_cast(value as float64) as price_value,
    cast(null as float64) as common_unit_price,
    cast(null as float64) as common_currency_price,
    year,
    case
        when safe_cast(months_code as int64) between 1 and 12
            then safe_cast(months_code as int64)
        else null
    end as month,
    months as faostat_months_label,
    cast(months_code as string) as faostat_months_code,
    safe_cast(area_code as int64) as area_code,
    safe_cast(area_code_m49 as int64) as area_code_m49,
    cast(null as string) as country_id,
    cast(null as string) as admin1_id,
    'faostat' as price_source,
    source_natural_key,
    current_timestamp() as loaded_at
from {{ ref('stg_faostat_prices') }}
where value is not null
  and safe_cast(value as float64) is not null
