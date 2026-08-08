{{ config(materialized='table') }}

-- Gold: emission category dimension from ClimateWatch emission pathways.

with categories as (
    select 'AGRI'      as category_code, 'Agriculture'                    as category_name, cast(null as string) as parent_category, 'All agricultural emission sources'       as description
    union all select 'ENERGY',           'Energy',                         null,             'Energy production and use'
    union all select 'LULUCF',           'Land Use, Land-Use Change & Forestry', null,      'Land use change and forestry'
    union all select 'WASTE',            'Waste',                          null,             'Waste management'
    union all select 'IPPU',             'Industrial Processes',           null,             'Industrial processes and product use'
    union all select 'ENTERIC',          'Enteric Fermentation',           'AGRI',           'Livestock enteric fermentation'
    union all select 'MANURE',           'Manure Management',              'AGRI',           'Livestock manure management'
    union all select 'RICE',             'Rice Cultivation',               'AGRI',           'Methane from paddy rice cultivation'
    union all select 'SYNTH_FERT',       'Synthetic Fertilizers',          'AGRI',           'N2O from synthetic fertilizer application'
    union all select 'CROP_RESID',       'Crop Residues',                  'AGRI',           'N2O from crop residue burning/decomposition'
)

select
    to_hex(md5(coalesce(category_code, '')))       as emission_category_key,
    category_code,
    category_name,
    parent_category,
    description

from categories
