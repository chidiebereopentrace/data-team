{{ config(materialized='table') }}

-- Gold: cross-border trade fact from FEWS cross-border trade time series.

select
    to_hex(md5(concat(
        coalesce(reporting_country_code, ''), '|',
        coalesce(product, ''), '|',
        coalesce(flow_type, ''), '|',
        coalesce(period_date, '')
    )))                                            as trade_key,

    to_hex(md5(concat(
        coalesce(reporting_country_code, ''), '|',
        coalesce(reporting_country, ''), '|', ''
    )))                                            as geo_key,

    cast(left(coalesce(period_date, '2000'), 4) as int64)
                                                   as time_key,

    to_hex(md5('FEWS_NET_cross_border_Trade_time_series_data'))
                                                   as source_key,

    to_hex(md5(coalesce(product, '')))             as product_key,

    to_hex(md5(concat(
        coalesce(flow_type, ''), '|', coalesce(trade_type, '')
    )))                                            as trade_flow_key,

    to_hex(md5(coalesce(unit, '')))                as unit_key,
    cast(null as string)                           as audit_key,
    cast(value as numeric)                         as value,
    cast(common_unit_quantity as numeric)          as common_unit_quantity,
    to_hex(md5(concat(
        coalesce(reporting_country_code, ''), '|',
        coalesce(product, ''), '|', coalesce(period_date, '')
    )))                                            as source_record_id

from {{ source('landing', 'FEWS_NET_cross_border_Trade_time_series_data') }}
where value is not null

{% if is_incremental() %}
  and period_date > (select cast(max(time_key) as string) from {{ this }})
{% endif %}
