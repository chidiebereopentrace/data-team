{{ config(materialized='table') }}

select
    to_hex(md5(src.item_name_norm)) as item_key,
    any_value(src.item_name) as item_name,
    any_value(src.item_code) as item_code,
    current_timestamp() as loaded_at
from (
    select item as item_name, cast(item_code as string) as item_code, lower(trim(item)) as item_name_norm
    from {{ ref('int_faostat_macro_conformed') }}
    where item is not null
    union all
    select item, cast(item_code as string), lower(trim(item))
    from {{ ref('int_employment_conformed') }}
    where item is not null
    union all
    select item, cast(item_code as string), lower(trim(item))
    from {{ ref('int_land_inputs_split') }}
    where item is not null
    union all
    select item, cast(item_code as string), lower(trim(item))
    from {{ ref('int_faostat_emissions_conformed') }}
    where item is not null
    union all
    select item, cast(item_code as string), lower(trim(item))
    from {{ ref('int_forestry_conformed') }}
    where item is not null
    union all
    select item, cast(item_code as string), lower(trim(item))
    from {{ ref('int_food_aid_conformed') }}
    where item is not null
    union all
    select item, cast(item_code as string), lower(trim(item))
    from {{ ref('int_gender_sdg_conformed') }}
    where item is not null
    union all
    select item, cast(item_code as string), lower(trim(item))
    from {{ ref('int_investment_asti_conformed') }}
    where item is not null
    union all
    select item, cast(item_code as string), lower(trim(item))
    from {{ ref('int_discontinued_machinery') }}
    where item is not null
    union all
    select product_name, cast(item_code as string), lower(trim(product_name))
    from {{ ref('int_faostat_trade_conformed') }}
    where product_name is not null
) src
group by src.item_name_norm
