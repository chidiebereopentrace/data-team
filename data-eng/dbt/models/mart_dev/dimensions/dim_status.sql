{{ config(materialized='table') }}

-- Gold: status dimension for collection and data quality states.

with statuses as (
    select 'ACTIVE'     as status_code, 'Active'            as status_name, 'collection' as status_type
    union all select 'INACTIVE',        'Inactive',          'collection'
    union all select 'PENDING',         'Pending',           'collection'
    union all select 'VALIDATED',       'Validated',         'quality'
    union all select 'FLAGGED',         'Flagged',           'quality'
    union all select 'REJECTED',        'Rejected',          'quality'
    union all select 'PUBLISHED',       'Published',         'publication'
    union all select 'DRAFT',           'Draft',             'publication'
)

select
    to_hex(md5(concat(coalesce(status_code, ''), '|', coalesce(status_type, ''))))
                                                   as status_key,
    status_code,
    status_name,
    status_type

from statuses
