{{ config(materialized='table') }}

select
    to_hex(md5(
        coalesce(g.geography_key, '') || '|' ||
        cast(h.year as string) || '|' ||
        coalesce(h.source_key, '')
    )) as agg_hdi_latest_key,
    g.geography_key,
    format_date('%Y%m%d', date(h.year, 1, 1)) as date_key,
    g.country_name,
    g.country_iso3,
    h.year as latest_year,
    h.value as hdi_value,
    h.source_key,
    current_timestamp() as loaded_at
from {{ ref('fct_hdi') }} h
left join {{ ref('dim_geography') }} g
    on g.geography_key = h.geography_key
qualify row_number() over (
    partition by g.country_iso3
    order by h.year desc
) = 1
