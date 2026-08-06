{{ config(materialized='table') }}

-- ILRI Dairy Genetics East Africa (DGEA1): milk yields + calving records.

select
    `Client ID for DGEA` as client_id,
    `Client name for DGEA` as client_name,
    Country as country,
    `Farm code` as farm_code,
    `Cow number` as cow_number,
    Breed as breed,
    `Number of lactations` as number_of_lactations,
    `Date of milking` as date_of_milking,
    `Test date` as test_date,
    `Yield Afternoon` as yield_afternoon,
    `Yield Morning` as yield_morning,
    cast(null as float64) as parity_number,
    cast(null as string) as initial_calving_date,
    'milk' as record_type,
    'ilri_dgea1_milkinfo_public' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'ilri_dgea1_milkinfo_public') }}

union all

select
    `Client ID for DGEA` as client_id,
    cast(null as string) as client_name,
    Country as country,
    `Farm code` as farm_code,
    `Cow number` as cow_number,
    cast(null as string) as breed,
    cast(null as float64) as number_of_lactations,
    cast(null as string) as date_of_milking,
    cast(null as string) as test_date,
    cast(null as float64) as yield_afternoon,
    cast(null as float64) as yield_morning,
    `Parity Number` as parity_number,
    `Initial calving date` as initial_calving_date,
    'calving' as record_type,
    'ilri_dgea1_calvinfo_public' as source_natural_key,
    current_timestamp() as loaded_at
from {{ source('raw_dev', 'ilri_dgea1_calvinfo_public') }}
