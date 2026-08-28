{% macro acf_data_level(geo_level_expr) %}
case {{ geo_level_expr }}
    when 'country' then 'national'
    when 'admin1' then 'sub_national'
    when 'admin2' then 'sub_national'
    when 'fnid' then 'sub_national'
    when 'city' then 'community'
    else 'point'
end
{% endmacro %}

{% macro acf_place_scope(prefix) %}
array(
    select distinct x
    from unnest([
        {{ prefix }}.country_iso3,
        {{ prefix }}.fnid,
        {{ prefix }}.admin_1_name,
        {{ prefix }}.admin_2_name,
        {{ prefix }}.city_name,
        {{ prefix }}.country_name
    ]) as x
    where x is not null and x != ''
)
{% endmacro %}

{# Dual geo join (native_id + ISO3 fallback); optional extra iso3 / country name exprs #}
{% macro acf_place_scope_coalesce(a, b, extra_iso3='cast(null as string)', extra_name='cast(null as string)') %}
array(
    select distinct x
    from unnest([
        coalesce({{ a }}.country_iso3, {{ b }}.country_iso3, {{ extra_iso3 }}),
        coalesce({{ a }}.fnid, {{ b }}.fnid),
        coalesce({{ a }}.admin_1_name, {{ b }}.admin_1_name),
        coalesce({{ a }}.admin_2_name, {{ b }}.admin_2_name),
        coalesce({{ a }}.city_name, {{ b }}.city_name),
        coalesce({{ a }}.country_name, {{ b }}.country_name, {{ extra_name }})
    ]) as x
    where x is not null and x != ''
)
{% endmacro %}

{% macro acf_row_data_level(geo_level_expr, default_data_level_expr) %}
coalesce(
    case {{ geo_level_expr }}
    when 'country' then 'national'
    when 'admin1' then 'sub_national'
    when 'admin2' then 'sub_national'
    when 'fnid' then 'sub_national'
    when 'city' then 'community'
    when 'point' then 'point'
    else null
    end,
    {{ default_data_level_expr }}
)
{% endmacro %}

{# Dim-only: map geo_level → data_level with no source-default fallback. #}
{% macro acf_row_data_level_strict(geo_level_expr) %}
case {{ geo_level_expr }}
    when 'country' then 'national'
    when 'admin1' then 'sub_national'
    when 'admin2' then 'sub_national'
    when 'fnid' then 'sub_national'
    when 'city' then 'community'
    when 'point' then 'point'
    else null
end
{% endmacro %}

{% macro acf_geo_scope(geo_level_expr, data_level_expr) %}
coalesce(
    case {{ geo_level_expr }}
        when 'country' then 'national'
        when 'admin1' then 'sub_national'
        when 'admin2' then 'sub_national'
        when 'fnid' then 'sub_national'
        when 'city' then 'community'
        when 'point' then 'point'
        else null
    end,
    case {{ data_level_expr }}
        when 'national' then 'national'
        when 'sub_national' then 'sub_national'
        when 'community' then 'community'
        when 'point' then 'point'
        else null
    end,
    'point'
)
{% endmacro %}

{# Dim-only geo_scope from geo_level only (null when geo unmatched). #}
{% macro acf_geo_scope_strict(geo_level_expr) %}
case {{ geo_level_expr }}
    when 'country' then 'national'
    when 'admin1' then 'sub_national'
    when 'admin2' then 'sub_national'
    when 'fnid' then 'sub_national'
    when 'city' then 'community'
    when 'point' then 'point'
    else null
end
{% endmacro %}

{# obs_date / year / month / loaded_at expressions; month must be 1-12 for month grain.
   Year must be 1-9999 (BigQuery DATE range) so year 0 cannot produce 0000-12-31. #}
{% macro acf_as_of_date(obs_date_expr='cast(null as date)', year_expr='cast(null as int64)', month_expr='cast(null as int64)', loaded_at_expr='current_timestamp()') %}
coalesce(
    {{ obs_date_expr }},
    case
        when {{ year_expr }} between 1 and 9999 and {{ month_expr }} between 1 and 12
            then date({{ year_expr }}, {{ month_expr }}, 1)
    end,
    case
        when {{ year_expr }} between 1 and 9999 then date({{ year_expr }}, 12, 31)
    end,
    date({{ loaded_at_expr }})
)
{% endmacro %}

{% macro acf_as_of_date_basis(obs_date_expr='cast(null as date)', year_expr='cast(null as int64)', month_expr='cast(null as int64)') %}
case
    when {{ obs_date_expr }} is not null then 'observation'
    when {{ year_expr }} between 1 and 9999 then 'observation'
    else 'loaded_at'
end
{% endmacro %}

{# Prefer dim ISO3; fallback to extracted ISO from intermediate / area map. #}
{% macro acf_country_iso3(dim_iso3_expr, fallback_iso3_expr='cast(null as string)') %}
coalesce({{ dim_iso3_expr }}, {{ fallback_iso3_expr }})
{% endmacro %}
