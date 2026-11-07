{{ config(materialized='table') }}

-- Gold: IPC/CH food security classification phases.
-- Sourced from staging_dev dim_ipc_phase; extended with phase_number and scale.

select
    ipc_phase_key                                  as classification_key,
    cast(ipc_phase_numeric as string)              as phase_code,
    ipc_phase_name                                 as phase_name,
    cast(ipc_phase_numeric as int64)               as phase_number,
    'IPC'                                          as classification_scale,
    case cast(ipc_phase_numeric as int64)
        when 1 then 'Minimal/None food insecurity'
        when 2 then 'Stressed'
        when 3 then 'Crisis'
        when 4 then 'Emergency'
        when 5 then 'Catastrophe/Famine'
        else ipc_phase_name
    end                                            as description

from {{ ref('dim_ipc_phase') }}
