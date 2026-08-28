{{ config(
    materialized='table',
    partition_by={'field': 'as_of_date', 'data_type': 'date', 'granularity': 'month'},
    cluster_by=['data_level', 'country_iso3', 'source_key', 'product_key']
) }}

{# Long-form food balance. Convenience columns map by element_code (not English labels).
   Food tonnes: 5141/5142; feed: 5520 (SUA) / 5521 (FBS); losses: 5016/5123;
   production: 5510/5511; import: 5610/5611; export: 5910/5911.
   Kcal / kg-capita elements stay in value only. #}

with {{ dim_country_by_native_id_cte() }},
{{ dim_country_by_iso3_cte() }},

base as (
    select
        to_hex(md5(
            coalesce(cast(b.area_code as string), '') || '|' ||
            coalesce(cast(b.item_code as string), '') || '|' ||
            coalesce(cast(b.element_code as string), '') || '|' ||
            coalesce(cast(b.year as string), '') || '|' ||
            coalesce(b.source_natural_key, '')
        )) as food_balance_key,
        coalesce(g_nat.geography_key, g_iso.geography_key) as geography_key,
        coalesce(g_nat.geo_level, g_iso.geo_level) as geo_level,
        {{ acf_country_iso3('coalesce(g_nat.country_iso3, g_iso.country_iso3)', 'iso.country_iso3') }} as country_iso3,
        pr.product_key,
        el.element_key,
        s.source_key,
        s.tier,
        {{ acf_row_data_level_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as data_level,
        {{ acf_geo_scope_strict('coalesce(g_nat.geo_level, g_iso.geo_level)') }} as geo_scope,
        {{ acf_place_scope_coalesce('g_nat', 'g_iso', 'iso.country_iso3', 'b.country_name') }} as place_scope,
        concat('food_balance_', lower(replace(coalesce(b.element, b.product_name, ''), ' ', '_'))) as metric,
        s.source_key as source_id,
        b.element,
        cast(b.element_code as string) as element_code,
        b.year,
        format_date('%Y%m%d', date(b.year, 1, 1)) as date_key,
        {{ acf_as_of_date('cast(null as date)', 'b.year', 'cast(null as int64)', 'b.loaded_at') }} as as_of_date,
        {{ acf_as_of_date_basis('cast(null as date)', 'b.year', 'cast(null as int64)') }} as as_of_date_basis,
        case when cast(b.element_code as string) in ('5510', '5511') then b.value end as production,
        case when cast(b.element_code as string) in ('5610', '5611') then b.value end as imports,
        case when cast(b.element_code as string) in ('5910', '5911') then b.value end as exports,
        case when cast(b.element_code as string) in ('5141', '5142') then b.value end as food,
        case when cast(b.element_code as string) in ('5520', '5521') then b.value end as feed,
        case when cast(b.element_code as string) in ('5016', '5123') then b.value end as losses,
        b.unit,
        b.value,
        b.source_natural_key,
        current_timestamp() as loaded_at
    from {{ ref('int_faostat_food_balances_conformed') }} b
    left join {{ ref('int_faostat_area_iso') }} iso
        on iso.area_code = cast(b.area_code as string)
    left join country_by_native_id g_nat
        on g_nat.native_id = cast(b.area_code as string)
    left join country_by_iso3 g_iso
        on upper(trim(g_iso.country_iso3)) = upper(trim(iso.country_iso3))
    left join {{ ref('dim_product') }} pr
        on pr.item_code = cast(b.item_code as string)
       and lower(pr.product_name) = lower(b.product_name)
    left join {{ ref('dim_element') }} el
        on lower(el.element_name) = lower(trim(b.element))
    left join {{ ref('dim_source') }} s
        on s.source_natural_key = b.source_natural_key
    where b.year is not null
)

select *
from base
qualify row_number() over (
    partition by food_balance_key
    order by value desc nulls last
) = 1
