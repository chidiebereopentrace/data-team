{{ config(materialized='table') }}

-- Gold: trade flow dimension for import/export/re-export classifications.

with flows as (
    select 'IMPORT'    as flow_type, 'Bilateral'  as trade_type, 'Imports into reporting country'          as description
    union all select 'EXPORT',       'Bilateral',               'Exports from reporting country'
    union all select 'REEXPORT',     'Bilateral',               'Re-exports from reporting country'
    union all select 'REIMPORT',     'Bilateral',               'Re-imports into reporting country'
    union all select 'CROSS_BORDER', 'Cross-Border',            'Informal cross-border trade flows'
)

select
    to_hex(md5(concat(coalesce(flow_type, ''), '|', coalesce(trade_type, ''))))
                                                   as trade_flow_key,
    flow_type,
    trade_type,
    description

from flows
