{{ config(materialized='table') }}

select
    to_hex(md5(src.indicator_name_norm)) as indicator_key,
    any_value(src.indicator_name) as indicator_name,
    current_timestamp() as loaded_at
from (
    select indicator as indicator_name, lower(trim(indicator)) as indicator_name_norm
    from {{ ref('int_employment_conformed') }}
    where indicator is not null
    union all
    select indicator, lower(trim(indicator))
    from {{ ref('int_climatewatch_conformed') }}
    where indicator is not null
    union all
    select series, lower(trim(series))
    from {{ ref('int_unccd_conformed') }}
    where series is not null
    union all
    select indicator, lower(trim(indicator))
    from {{ ref('int_investment_asti_conformed') }}
    where indicator is not null
) src
where src.indicator_name is not null
group by src.indicator_name_norm
