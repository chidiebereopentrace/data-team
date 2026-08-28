{{ config(materialized='table') }}

select
    to_hex(md5(src.element_name_norm)) as element_key,
    any_value(src.element_name) as element_name,
    any_value(src.element_code) as element_code,
    current_timestamp() as loaded_at
from (
    select element as element_name, cast(element_code as string) as element_code, lower(trim(element)) as element_name_norm
    from {{ ref('int_faostat_macro_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_employment_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_land_inputs_split') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_faostat_emissions_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_forestry_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_food_aid_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_gender_sdg_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_investment_asti_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_discontinued_machinery') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_faostat_trade_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_faostat_production_conformed') }}
    where element is not null
    union all
    select element, cast(element_code as string), lower(trim(element))
    from {{ ref('int_faostat_food_balances_conformed') }}
    where element is not null
) src
group by src.element_name_norm
