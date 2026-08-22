{{ config(materialized='table') }}

-- ILRI Index-Based Livestock Insurance: farmer survey + vaccinator sampling.

with base as (
    select
        cast(HouseholdID as string) as household_id,
        country,
        cast(RegionID as string) as region_id,
        FarmerLocation as farmer_location,
        farmer_location_new,
        FarmerCat1 as farmer_category,
        Categorisation as categorisation,
        HerdSizeCat as herd_size_category,
        herd_size_cat,
        herd_size_cat_new,
        ItmStartYear as insurance_start_year,
        itm_start_year,
        ItmStartYearNew as insurance_start_year_new,
        cast(VaccID as string) as vaccinator_id,
        local_currency,
        currency_label,
        'farmer' as record_type,
        'ilri_i4i_farmerdataanon' as source_natural_key
    from {{ source('raw_dev', 'ilri_i4i_farmerdataanon') }}

    union all

    select
        cast(null as string) as household_id,
        cast(null as string) as country,
        cast(VaccRegionID as string) as region_id,
        cast(null as string) as farmer_location,
        cast(null as string) as farmer_location_new,
        cast(null as string) as farmer_category,
        cast(null as float64) as categorisation,
        cast(null as string) as herd_size_category,
        cast(null as float64) as herd_size_cat,
        cast(null as string) as herd_size_cat_new,
        cast(null as float64) as insurance_start_year,
        cast(null as float64) as itm_start_year,
        cast(null as float64) as insurance_start_year_new,
        cast(VaccID as string) as vaccinator_id,
        cast(null as string) as local_currency,
        cast(null as string) as currency_label,
        'vaccinator' as record_type,
        'ilri_i4i_samplingdata_anon' as source_natural_key
    from {{ source('raw_dev', 'ilri_i4i_samplingdata_anon') }}
)

select
    to_hex(md5(to_json_string(base))) as ilri_i4i_livestock_insurance_sk,
    base.*,
    current_timestamp() as loaded_at
from base
