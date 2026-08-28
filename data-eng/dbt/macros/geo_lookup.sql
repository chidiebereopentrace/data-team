{% macro geo_africa_bbox_filter(lat_expr, lon_expr) %}
(
    {{ lon_expr }} between -25 and 60
    and {{ lat_expr }} between -35 and 38
)
{% endmacro %}

{# Shared Africa ISO2 allowlist — source of truth is ref_m49_country.in_africa_scope. #}
{% macro geo_africa_iso2_codes() %}
{{ return([
  'DZ','AO','BJ','BW','BF','BI','CM','CV','CF','TD','KM','CG','CD','CI','DJ',
  'EG','GQ','ER','SZ','ET','GA','GM','GH','GN','GW','KE','LS','LR','LY','MG',
  'MW','ML','MR','MU','MA','MZ','NA','NE','NG','RW','ST','SN','SC','SL','SO',
  'ZA','SS','SD','TZ','TG','TN','UG','ZM','ZW',
  'YT','RE','SH','EH','TF'
]) }}
{% endmacro %}

{% macro geo_africa_iso2_in(expr) %}
{{ expr }} in (
    select country_iso2 from {{ ref('ref_m49_country') }}
    where in_africa_scope
)
{% endmacro %}

{# Arabian / Yemen cells inside the Africa rectangle but outside Africa land. #}
{% macro geo_africa_soil_fringe_exclude(lat_expr, lon_expr) %}
not ({{ lat_expr }} > 12.5 and {{ lon_expr }} > 43)
{% endmacro %}

{# FAOSTAT area name → ISO3 when M49 / ref country name miss (territories + aliases). #}
{% macro geo_faostat_country_name_iso3(country_name_expr) %}
case
    when lower({{ country_name_expr }}) like '%ivoire%'
      or lower({{ country_name_expr }}) like '%ivory coast%' then 'CIV'
    when lower({{ country_name_expr }}) like '%tanzania%' then 'TZA'
    when lower({{ country_name_expr }}) like '%eswatini%'
      or lower({{ country_name_expr }}) like '%swaziland%' then 'SWZ'
    when lower({{ country_name_expr }}) like '%congo%democratic%'
      or lower({{ country_name_expr }}) like '%dr congo%'
      or lower({{ country_name_expr }}) = 'democratic republic of the congo' then 'COD'
    when lower({{ country_name_expr }}) like '%congo%'
      and lower({{ country_name_expr }}) not like '%democratic%' then 'COG'
    when lower({{ country_name_expr }}) like '%cabo verde%'
      or lower({{ country_name_expr }}) like '%cape verde%' then 'CPV'
    when lower({{ country_name_expr }}) like '%gambia%' then 'GMB'
    when lower({{ country_name_expr }}) like '%guinea-bissau%' then 'GNB'
    when lower({{ country_name_expr }}) like '%sao tome%'
      or lower({{ country_name_expr }}) like '%são tomé%' then 'STP'
    when lower({{ country_name_expr }}) like '%sudan%'
      and lower({{ country_name_expr }}) like '%south%' then 'SSD'
    when lower({{ country_name_expr }}) like '%sudan%'
      and lower({{ country_name_expr }}) like '%former%' then 'SDN'
    when lower({{ country_name_expr }}) = 'sudan' then 'SDN'
    when lower({{ country_name_expr }}) like '%ethiopia%pdr%'
      or lower({{ country_name_expr }}) like '%ethiopia pdr%' then 'ETH'
    when lower({{ country_name_expr }}) like '%mayotte%' then 'MYT'
    when lower({{ country_name_expr }}) like '%reunion%'
      or lower({{ country_name_expr }}) like '%réunion%'
      or lower({{ country_name_expr }}) like '%runion%' then 'REU'
    when lower({{ country_name_expr }}) like '%western sahara%' then 'ESH'
    when lower({{ country_name_expr }}) like '%saint helena%'
      or lower({{ country_name_expr }}) like '%st helena%' then 'SHN'
    when lower({{ country_name_expr }}) like '%french southern%'
      or lower({{ country_name_expr }}) like '%french southern and antarctic%' then 'ATF'
end
{% endmacro %}

{# Normalize FAOSTAT Area_Code_M49 (strip leading quotes) then zero-pad. #}
{% macro geo_faostat_m49_norm(expr) %}
lpad(regexp_replace(trim(safe_cast({{ expr }} as string)), r"^'+", ''), 3, '0')
{% endmacro %}

{# Africa cities for nearest-city attach (ST_DISTANCE candidates). #}
{% macro geo_africa_cities_cte(cte_name='africa_cities') %}
{{ cte_name }} as (
    select
        geo_key,
        latitude,
        longitude,
        upper(trim(country_iso2)) as country_iso2,
        upper(trim(country_iso3)) as country_iso3,
        country_name,
        city_name,
        population
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'city'
      and latitude is not null
      and longitude is not null
      and country_iso2 is not null
      and {{ geo_africa_bbox_filter('latitude', 'longitude') }}
)
{% endmacro %}

{# Centroid lat/lon from WKT; null when WKT missing/unparseable.
   SAFE only on ST_GEOGFROMTEXT — SAFE.ST_Y / SAFE.ST_X are not valid in BigQuery. #}
{% macro geo_centroid_latitude(wkt_expr) %}
st_y(st_centroid(safe.st_geogfromtext({{ wkt_expr }})))
{% endmacro %}

{% macro geo_centroid_longitude(wkt_expr) %}
st_x(st_centroid(safe.st_geogfromtext({{ wkt_expr }})))
{% endmacro %}

{# Centroid from a GEOGRAPHY-typed column. #}
{% macro geo_geog_centroid_latitude(geog_expr) %}
st_y(st_centroid({{ geog_expr }}))
{% endmacro %}

{% macro geo_geog_centroid_longitude(geog_expr) %}
st_x(st_centroid({{ geog_expr }}))
{% endmacro %}

{# Resolve geo_key: exact city → nearest city within max_m → country key. #}
{% macro geo_resolve_point_geo_key(exact_key, nearest_city_key, nearest_dist_m, country_key, max_m=100000) %}
coalesce(
    {{ exact_key }},
    case
        when {{ nearest_dist_m }} is not null and {{ nearest_dist_m }} <= {{ max_m }}
            then {{ nearest_city_key }}
    end,
    {{ country_key }}
)
{% endmacro %}

{% macro geo_country_by_iso2_cte(cte_name='iso2_geo') %}
{{ cte_name }} as (
    select
        upper(trim(country_iso2)) as country_iso2,
        geo_key,
        country_iso3,
        country_name
    from {{ ref('int_geography_conformed') }}
    where country_iso2 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso2))
        order by
            case geo_level when 'country' then 0 else 1 end,
            case when capital_status is not null then 0 else 1 end,
            geo_key
    ) = 1
)
{% endmacro %}

{% macro geo_country_by_name_cte(cte_name='country_by_name') %}
{{ cte_name }} as (
    select
        lower(trim(country_name)) as country_name_norm,
        geo_key,
        country_iso2,
        country_iso3,
        country_name
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'country'
      and country_name is not null
    qualify row_number() over (
        partition by lower(trim(country_name))
        order by population desc nulls last, geo_key
    ) = 1
)
{% endmacro %}

{% macro geo_city_by_latlon_cte(cte_name='city_by_latlon') %}
{{ cte_name }} as (
    select
        round(latitude, 2) as lat_round,
        round(longitude, 2) as lon_round,
        geo_key,
        geo_level,
        country_iso2,
        country_iso3,
        country_name,
        city_name
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'city'
      and latitude is not null
      and longitude is not null
    qualify row_number() over (
        partition by cast(round(latitude, 2) as string), cast(round(longitude, 2) as string)
        order by population desc nulls last, geo_key
    ) = 1
)
{% endmacro %}

{% macro geo_city_latlon_join(point_alias, city_alias) %}
round({{ point_alias }}.latitude, 2) = {{ city_alias }}.lat_round
and round({{ point_alias }}.longitude, 2) = {{ city_alias }}.lon_round
{% endmacro %}

{# Bbox prefilter for nearest-city candidates (~1.2 deg ≈ 130km). #}
{% macro geo_nearest_city_bbox_join(point_alias, city_alias, bbox_deg=1.2) %}
{{ point_alias }}.latitude is not null
and {{ point_alias }}.longitude is not null
and abs({{ point_alias }}.latitude - {{ city_alias }}.latitude) <= {{ bbox_deg }}
and abs({{ point_alias }}.longitude - {{ city_alias }}.longitude) <= {{ bbox_deg }}
{% endmacro %}

{% macro geo_st_distance_m(point_alias, city_alias) %}
st_distance(
    st_geogpoint({{ point_alias }}.longitude, {{ point_alias }}.latitude),
    st_geogpoint({{ city_alias }}.longitude, {{ city_alias }}.latitude)
)
{% endmacro %}

{% macro geo_fews_admin2_join(row_alias, geo_alias) %}
{{ geo_alias }}.geo_level = 'admin2'
and upper(trim({{ geo_alias }}.country_iso2)) = upper(trim({{ row_alias }}.country_code))
and coalesce({{ geo_alias }}.admin_1_name, '') = coalesce({{ row_alias }}.admin_1, '')
and coalesce({{ geo_alias }}.admin_2_name, '') = coalesce({{ row_alias }}.admin_2, '')
{% endmacro %}

{% macro geo_fews_admin1_join(row_alias, geo_alias) %}
{{ geo_alias }}.geo_level = 'admin1'
and upper(trim({{ geo_alias }}.country_iso2)) = upper(trim({{ row_alias }}.country_code))
and coalesce({{ geo_alias }}.admin_1_name, '') = coalesce({{ row_alias }}.admin_1, '')
{% endmacro %}

{% macro geo_fews_country_join(row_alias, geo_alias) %}
{{ geo_alias }}.geo_level = 'country'
and (
    upper(trim({{ geo_alias }}.country_iso2)) = upper(trim({{ row_alias }}.country_code))
    or lower(trim({{ geo_alias }}.country_name)) = lower(trim({{ row_alias }}.country))
)
{% endmacro %}

{% macro geo_wfp_admin1_join(row_alias, geo_alias) %}
{{ geo_alias }}.geo_level = 'admin1'
and lower(trim({{ geo_alias }}.country_name)) = lower(trim({{ row_alias }}.country))
and coalesce({{ geo_alias }}.admin_1_name, '') = coalesce({{ row_alias }}.admin_1, '')
{% endmacro %}

{% macro geo_wfp_admin1_join_by_iso3(row_alias, geo_alias, iso3_expr) %}
{{ geo_alias }}.geo_level = 'admin1'
and upper(trim({{ geo_alias }}.country_iso3)) = upper(trim({{ iso3_expr }}))
and coalesce({{ geo_alias }}.admin_1_name, '') = coalesce({{ row_alias }}.admin_1, '')
{% endmacro %}

{% macro geo_country_by_iso3_cte(cte_name='country_by_iso3') %}
{{ cte_name }} as (
    select
        upper(trim(country_iso3)) as country_iso3_norm,
        geo_key,
        country_iso2,
        country_iso3,
        country_name
    from {{ ref('int_geography_conformed') }}
    where geo_level = 'country'
      and country_iso3 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso3))
        order by population desc nulls last, geo_key
    ) = 1
)
{% endmacro %}

{% macro dim_country_by_native_id_cte(cte_name='country_by_native_id') %}
{{ cte_name }} as (
    select *
    from {{ ref('dim_geography') }}
    where geo_level = 'country'
      and native_id is not null
    qualify row_number() over (
        partition by native_id
        order by population desc nulls last, geography_key
    ) = 1
)
{% endmacro %}

{% macro dim_country_by_iso3_cte(cte_name='country_by_iso3') %}
{{ cte_name }} as (
    select *
    from {{ ref('dim_geography') }}
    where geo_level = 'country'
      and country_iso3 is not null
    qualify row_number() over (
        partition by upper(trim(country_iso3))
        order by population desc nulls last, geography_key
    ) = 1
)
{% endmacro %}
