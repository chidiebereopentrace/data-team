{# Unified geography resolution — point grid, FAOSTAT area, ref country. #}

{% macro geo_latlon_grid_cte(cte_name='latlon_grid') %}
{{ cte_name }} as (
    select
        lat_cell,
        lon_cell,
        country_iso2,
        country_iso3,
        country_name
    from {{ ref('int_latlon_country_grid') }}
)
{% endmacro %}

{% macro geo_latlon_grid_join(point_alias, grid_alias) %}
round({{ point_alias }}.latitude, 1) = {{ grid_alias }}.lat_cell
and round({{ point_alias }}.longitude, 1) = {{ grid_alias }}.lon_cell
{% endmacro %}

{% macro geo_resolve_country_iso3_from_grid(grid_iso3_expr, grid_iso2_expr, fallback_iso3_expr='cast(null as string)') %}
coalesce({{ grid_iso3_expr }}, {{ fallback_iso3_expr }})
{% endmacro %}

{% macro geo_faostat_area_cte(cte_name='faostat_area') %}
{{ cte_name }} as (
    select
        area_code,
        area_code_m49,
        country_name,
        country_iso2,
        country_iso3,
        match_method
    from {{ ref('int_faostat_area_reference') }}
)
{% endmacro %}

{% macro geo_ref_country_by_iso2_cte(cte_name='ref_country_iso2') %}
{{ cte_name }} as (
    select
        country_iso2,
        country_iso3,
        country_name,
        in_africa_scope
    from {{ ref('int_ref_country') }}
    where country_iso2 is not null
)
{% endmacro %}

{# Point row: grid country + optional nearest city for community geo_key. #}
{% macro geo_resolve_point_country_from_grid(point_alias, grid_alias) %}
coalesce(
    {{ grid_alias }}.country_iso3,
    cast(null as string)
)
{% endmacro %}

{% macro geo_country_geo_key_by_iso2_cte(cte_name='country_geo_by_iso2', geography_ref='int_geography_conformed') %}
{{ cte_name }} as (
    select
        upper(trim(country_iso2)) as country_iso2,
        geo_key,
        country_iso3,
        country_name
    from {{ ref(geography_ref) }}
    where geo_level = 'country'
      and country_iso2 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso2))
        order by population desc nulls last, geo_key
    ) = 1
)
{% endmacro %}

{# Standard point resolution: exact city → nearest city → grid country → dim country. #}
{% macro geo_coalesce_point_country_iso2(exact_iso2, nearest_dist_m, nearest_iso2, grid_iso2, country_iso2_expr, fallback_nearest_iso2) %}
coalesce(
    {{ exact_iso2 }},
    case when {{ nearest_dist_m }} <= 100000 then {{ nearest_iso2 }} end,
    {{ grid_iso2 }},
    {{ country_iso2_expr }},
    {{ fallback_nearest_iso2 }}
)
{% endmacro %}

{% macro geo_coalesce_point_country_iso3(exact_iso3, nearest_dist_m, nearest_iso3, grid_iso3, country_iso3_expr, fallback_nearest_iso3) %}
coalesce(
    {{ exact_iso3 }},
    case when {{ nearest_dist_m }} <= 100000 then {{ nearest_iso3 }} end,
    {{ grid_iso3 }},
    {{ country_iso3_expr }},
    {{ fallback_nearest_iso3 }}
)
{% endmacro %}

{% macro geo_iso2_for_country_dim_join(exact_iso2, nearest_dist_m, nearest_iso2, grid_iso2, fallback_nearest_iso2, admin0_iso2='cast(null as string)') %}
coalesce(
    {{ exact_iso2 }},
    case when {{ nearest_dist_m }} <= 100000 then {{ nearest_iso2 }} end,
    {{ grid_iso2 }},
    {{ admin0_iso2 }},
    {{ fallback_nearest_iso2 }}
)
{% endmacro %}

{% macro geo_admin0_africa_cte(cte_name='admin0_africa') %}
{{ cte_name }} as (
    select
        country_iso2,
        country_iso3,
        country_name,
        min_lat,
        max_lat,
        min_lng,
        max_lng,
        geog,
        source_priority
    from {{ ref('int_admin0_africa') }}
)
{% endmacro %}

{% macro geo_admin0_bbox_prefilter(point_alias, admin_alias) %}
{{ point_alias }}.latitude between {{ admin_alias }}.min_lat and {{ admin_alias }}.max_lat
and {{ point_alias }}.longitude between {{ admin_alias }}.min_lng and {{ admin_alias }}.max_lng
{% endmacro %}

{% macro geo_admin0_contains_join(point_alias, admin_alias) %}
st_contains(
    {{ admin_alias }}.geog,
    st_geogpoint({{ point_alias }}.longitude, {{ point_alias }}.latitude)
)
{% endmacro %}
