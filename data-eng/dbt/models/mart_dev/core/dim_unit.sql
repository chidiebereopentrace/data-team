{{ config(materialized='table') }}

select
    to_hex(md5(src.unit_code_norm || '|' || src.unit_type)) as unit_key,
    any_value(src.unit_code) as unit_code,
    src.unit_type,
    current_timestamp() as loaded_at
from (
    select unit as unit_code, 'quantity' as unit_type, lower(trim(unit)) as unit_code_norm
    from {{ ref('int_faostat_production_conformed') }}
    where unit is not null
    union all
    select unit, 'quantity', lower(trim(unit))
    from {{ ref('int_prices_harmonised') }}
    where unit is not null
    union all
    select currency, 'currency', lower(trim(currency))
    from {{ ref('int_prices_harmonised') }}
    where currency is not null
    union all
    select 'tonnes' as unit_code, 'quantity' as unit_type, 'tonnes' as unit_code_norm
    union all
    select 'ha', 'quantity', 'ha'
    union all
    select 't/ha', 'quantity', 't/ha'
) src
where src.unit_code is not null
group by
    src.unit_code_norm,
    src.unit_type
