{{ config(materialized='table') }}

-- Gold aggregate: monthly food security summary by country and IPC phase.
-- Pre-aggregated for dashboard performance (Power BI / Looker).

select
    f.geo_key,
    g.country_code,
    g.country_name,
    f.time_key,
    f.classification_key,
    c.phase_name,
    c.phase_number,
    f.scenario_key,
    s.scenario_name,
    count(*)                                       as record_count,
    sum(f.value)                                   as total_population_affected,
    avg(f.value)                                   as avg_population_affected,
    avg(f.pct_phase3)                              as avg_pct_phase3,
    avg(f.pct_phase4)                              as avg_pct_phase4,
    avg(f.pct_phase5)                              as avg_pct_phase5

from {{ ref('fct_food_security') }} f
left join {{ ref('gold_dim_geography') }}       g on f.geo_key = g.geography_key
left join {{ ref('dim_classification') }}  c on f.classification_key = c.classification_key
left join {{ ref('dim_scenario') }}        s on f.scenario_key = s.scenario_key

group by 1, 2, 3, 4, 5, 6, 7, 8, 9
