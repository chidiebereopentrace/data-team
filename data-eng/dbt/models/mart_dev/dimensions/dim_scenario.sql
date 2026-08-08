{{ config(materialized='table') }}

-- Gold: scenario dimension for FEWS food security projection scenarios.
-- CS = Current Situation, ML = Most Likely, OW = Optimistic/Worst-case.

with scenarios as (
    select 'CS' as scenario_code, 'Current Situation'   as scenario_name, 'IPC current-period scenario'                    as description
    union all select 'ML',        'Most Likely',          'IPC projected most-likely scenario'
    union all select 'OW',        'Optimistic/Worst',     'IPC projected optimistic or worst-case scenario'
    union all select 'BAU',       'Business As Usual',    'Baseline no-intervention trajectory'
    union all select 'PROJ',      'Projection',           'Forward-looking scenario'
)

select
    to_hex(md5(coalesce(scenario_code, '')))       as scenario_key,
    scenario_code,
    scenario_name,
    description,
    true                                           as is_current

from scenarios
