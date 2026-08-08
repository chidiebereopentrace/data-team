{{ config(materialized='table') }}

-- Gold: links research projects to their implementing organisations.
-- Sourced from OpenAire project-organisation linkages.

-- Product_links_bronze has: openaire_id, rel_name, target_id, target_type.
-- No direct project_id/organisation_id columns — bridge populated once
-- a join model mapping openaire_id -> project/org is built.

select
    cast(null as string)                           as research_project_key,
    cast(null as string)                           as organisation_key

from (select 1 as _dummy)
where false
