# PRD: data-eng transform-v2 reorganization

Status: ready-for-agent (local — publish to GitHub when `gh` is available)

## Problem Statement

Data-eng had competing mental models (Airbyte vs Python ingest, bronze/silver/gold vs `*_dev`, dbt owning landing→raw). Engineers could not follow one path from load to mart without risking the working dbt setup.

## Solution

Cut **transform-v2**: ingestion writes `landing_dev` and `raw_dev`; dbt treats `raw_dev` as the primordial source through `mart_dev`. Soft cutover — no rewrite of existing dbt model SQL. Airbyte removed from the active tree. Generated source catalog under `dbt/catalog/` (committed). Pipeline versions via git tags `transform-v1.0.0` / `transform-v2.*`.

## User Stories

1. As a data engineer, I want one README ownership story, so that I know whether ingest or dbt writes `raw_dev`.
2. As a data engineer, I want Airbyte out of the default path, so that I do not configure a dead stack.
3. As a data engineer, I want transform-v1 recoverable by tag, so that I can roll back the contract cut.
4. As a data engineer, I want transform-v2 tagged and versioned in `dbt_project`, so that releases are explicit.
5. As a data engineer, I want the mega `sources.yml` out of `models/`, so that the IDE and agents stay usable.
6. As a data engineer, I want to regenerate the source catalog after clone, so that dbt still has table lists.
7. As a data engineer, I want existing staging/intermediate/mart SQL untouched, so that marts keep working.
8. As a data engineer, I want transitional landing/raw_dev dbt runners left in place, so that soft cutover does not break loads.
9. As a data engineer, I want CONTEXT.md glossary terms, so that bronze/silver/gold are not team language.
10. As a data engineer, I want ADRs for ingest-only, ownership, and catalog placement, so that decisions stick.
11. As a platform engineer, I want Airbyte workflows and DAGs archived, so that CI does not target a retired path.
12. As a new teammate, I want first-time setup to write `catalog/sources.generated.yml`, so that onboarding matches the contract.
13. As a data engineer, I want new sources to go through ingestion into `landing_dev`/`raw_dev`, so that ownership stays clear.
14. As a data engineer, I want local path rename deferred if risky, so that compose/scripts do not break.
15. As an agent, I want follow-up issues for remaining slices, so that work continues without a big bang.

## Implementation Decisions

- Same `data-eng/dbt/` tree; no parallel `dbt-v2/` copy.
- v2.0.0 is contract + docs + versioning; model SQL identical to v1 at cut.
- Ingest documents writing `landing_dev` and `raw_dev`; per-source raw writes can land later.
- Generated catalog: `dbt/catalog/sources.generated.yml`; `model-paths` includes `catalog`.
- Physical rename `data/local` → `local` deferred.
- Terraform Airbyte modules marked retired for new work.

## Testing Decisions

- Contract cut does not require a full mart rebuild.
- Verify tags, `dbt_project` version `2.0.0`, catalog script output path, no active `data-eng/airbyte/`.

## Out of Scope

- Rewriting staging/intermediate/mart SQL
- Big-bang delete of `dbt/models/landing` or `models/raw_dev`
- Production `*_prod` / RAG contract changes

## Further Notes

- Session: `.agent/context/architecture-sessions/2026-10-17-data-eng-reorg.md`
- ADRs: `data-eng/docs/adr/0001`–`0003`
- Tags: `transform-v1.0.0`, `transform-v2.0.0`
