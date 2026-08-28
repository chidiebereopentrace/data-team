{{ config(materialized='table') }}

with keys as (
    select source_natural_key from {{ ref('stg_fews_food_security') }}
    union distinct select source_natural_key from {{ ref('stg_fews_market_prices') }}
    union distinct select source_natural_key from {{ ref('stg_fews_cross_border_trade') }}
    union distinct select source_natural_key from {{ ref('stg_fews_data_series_catalogue') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_production') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_macro') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_prices') }}
    union distinct select source_natural_key from {{ ref('int_prices_harmonised') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_trade') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_land_inputs') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_food_balances') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_emissions') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_forestry') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_food_aid') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_discontinued_machinery') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_investment_asti') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_sdg_hdi') }}
    union distinct select source_natural_key from {{ ref('stg_faostat_population_employment') }}
    union distinct select source_natural_key from {{ ref('stg_wfp_vampire_prices') }}
    union distinct select source_natural_key from {{ ref('stg_yield_raw_data') }}
    union distinct select source_natural_key from {{ ref('stg_africa_hdi') }}
    union distinct select source_natural_key from {{ ref('stg_africa_gdp_ppp') }}
    union distinct select source_natural_key from {{ ref('stg_isda_soil_enriched') }}
    union distinct select source_natural_key from {{ ref('stg_isric_africa_soil') }}
    union distinct select source_natural_key from {{ ref('stg_unccd_land_degradation') }}
    union distinct select source_natural_key from {{ ref('stg_s4a_field_surveys') }}
    union distinct select source_natural_key from {{ ref('stg_cifor_icraf') }}
    union distinct select source_natural_key from {{ ref('stg_nasa_power') }}
    union distinct select source_natural_key from {{ ref('stg_copernicus_era5') }}
    union distinct select source_natural_key from {{ ref('stg_climatewatch_health') }}
    union distinct select source_natural_key from {{ ref('stg_nakuru_air_quality') }}
    union distinct select source_natural_key from {{ ref('stg_un_peace_security') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_household_food_security') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_animal_health') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_food_hazards') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_i4i_livestock_insurance') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_vegetation_feed') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_vendor_consumer') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_other_surveys') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_dairy_genetics') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_genomics') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_buffel_silicosdart_markers') }}
    union distinct select source_natural_key from {{ ref('stg_ilri_buffel_snp_markers') }}
    union distinct select source_natural_key from {{ ref('stg_biodiversity') }}
    union distinct select source_natural_key from {{ ref('stg_protected_areas') }}
    union distinct select source_natural_key from {{ ref('stg_germplasm') }}
    union distinct select source_natural_key from {{ ref('stg_vegetation_ndvi') }}
    union distinct select source_natural_key from {{ ref('stg_aez') }}
    union distinct select source_natural_key from {{ ref('stg_arcgis_poi') }}
    union distinct select source_natural_key from {{ ref('stg_arcgis_sa_wards_demographics') }}
    union distinct select source_natural_key from {{ ref('stg_openaire_projects') }}
    union distinct select source_natural_key from {{ ref('stg_openaire_organisations') }}
    union distinct select source_natural_key from {{ ref('stg_openaire_persons') }}
    union distinct select source_natural_key from {{ ref('stg_openaire_product_links') }}
    union distinct select source_natural_key from {{ ref('stg_openaire_data_sources') }}
)

select
    source_natural_key,
    case
        when source_natural_key like 'FEWS%' then 'FEWS NET'
        when source_natural_key like 'Unpivoted_FAOstat%' then 'FAOSTAT'
        when lower(source_natural_key) like '%vampire%' or source_natural_key like 'WFP%' then 'WFP'
        when source_natural_key like 'ilri%' then 'ILRI'
        when lower(source_natural_key) like '%s4a%' then 'S4A field surveys'
        when lower(source_natural_key) like '%cifor%' then 'CIFOR-ICRAF'
        when lower(source_natural_key) like '%isric%' then 'ISRIC'
        when lower(source_natural_key) like '%isda%' then 'iSDA'
        when lower(source_natural_key) like '%openaire%' then 'OpenAIRE'
        when lower(source_natural_key) like '%climatewatch%' then 'Climate Watch'
        when lower(source_natural_key) like '%era5%' or lower(source_natural_key) like '%copernicus%' then 'Copernicus / ERA5'
        when lower(source_natural_key) like '%nasa%' or lower(source_natural_key) like '%power%' then 'NASA POWER'
        when lower(source_natural_key) like '%unccd%' then 'UNCCD'
        when lower(source_natural_key) like '%gbif%' or lower(source_natural_key) like '%biodiversity%' then 'GBIF / biodiversity'
        when source_natural_key like '%Human_development%' then 'UNDP / HDI extract'
        when source_natural_key like '%gdp%' or source_natural_key like '%GDP%' then 'World Bank / GDP extract'
        when source_natural_key = 'yield_raw_data' then 'Yield extract'
        else 'other'
    end as organisation_name,
    case
        when source_natural_key like 'FEWS%' then 1
        when source_natural_key like 'Unpivoted_FAOstat%' then 1
        when lower(source_natural_key) like '%vampire%' or source_natural_key like 'WFP%' then 1
        when lower(source_natural_key) like '%nasa%' or lower(source_natural_key) like '%power%' then 1
        when lower(source_natural_key) like '%era5%' or lower(source_natural_key) like '%copernicus%' then 1
        when lower(source_natural_key) like '%isric%' or lower(source_natural_key) like '%isda%' then 1
        when source_natural_key like '%Human_development%' then 1
        when source_natural_key like '%gdp%' or source_natural_key like '%GDP%' then 1
        when lower(source_natural_key) like '%climatewatch%' then 1
        when lower(source_natural_key) like '%gbif%' or lower(source_natural_key) like '%biodiversity%' then 1
        when source_natural_key like 'ilri%' then 3
        when lower(source_natural_key) like '%s4a%' then 3
        when lower(source_natural_key) like '%cifor%' then 3
        else 2
    end as tier,
    case
        when source_natural_key like 'FEWS%' then 'sub_national'
        when source_natural_key like 'ilri%' then 'community'
        when lower(source_natural_key) like '%s4a%' then 'community'
        when lower(source_natural_key) like '%cifor%' then 'community'
        when lower(source_natural_key) like '%era5%' or lower(source_natural_key) like '%copernicus%' then 'point'
        when lower(source_natural_key) like '%nasa%' or lower(source_natural_key) like '%power%' then 'point'
        when lower(source_natural_key) like '%isric%' or lower(source_natural_key) like '%isda%' then 'point'
        when lower(source_natural_key) like '%gbif%' or lower(source_natural_key) like '%biodiversity%' then 'point'
        when lower(source_natural_key) like '%nakuru%' or lower(source_natural_key) like '%air_quality%' then 'point'
        when lower(source_natural_key) like '%ndvi%' or lower(source_natural_key) like '%vegetation%' then 'point'
        else 'national'
    end as default_data_level,
    case
        when source_natural_key like 'FEWS%' then 'regional'
        when source_natural_key like 'Unpivoted_FAOstat%' then 'global'
        when lower(source_natural_key) like '%vampire%' or source_natural_key like 'WFP%' then 'global'
        when lower(source_natural_key) like '%nasa%' or lower(source_natural_key) like '%power%' then 'global'
        when lower(source_natural_key) like '%era5%' or lower(source_natural_key) like '%copernicus%' then 'global'
        when lower(source_natural_key) like '%isric%' or lower(source_natural_key) like '%isda%' then 'global'
        when source_natural_key like '%Human_development%' then 'global'
        when source_natural_key like '%gdp%' or source_natural_key like '%GDP%' then 'global'
        when lower(source_natural_key) like '%climatewatch%' then 'global'
        when lower(source_natural_key) like '%gbif%' or lower(source_natural_key) like '%biodiversity%' then 'global'
        when lower(source_natural_key) like '%openaire%' then 'global'
        when source_natural_key = 'yield_raw_data' then 'regional'
        when source_natural_key like 'ilri%' then 'subnational'
        when lower(source_natural_key) like '%s4a%' then 'subnational'
        when lower(source_natural_key) like '%cifor%' then 'subnational'
        when lower(source_natural_key) like '%nakuru%' then 'subnational'
        else 'national'
    end as producer_scale,
    current_timestamp() as loaded_at
from keys
where source_natural_key is not null
