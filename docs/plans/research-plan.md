# Research & Academia plan and perspective

## Scope

Add the paid `Research` plan with display name **Research & Academia**. The canonical API values for both `user_profile.plan_type` and `user_profile.category` are `Research`; the plan URL slug is `research`. The obsolete proposed display name Research Institutions & Academia is not used.

- `POST /query/research` uses the same request, response, authentication, metering, session handling and trace controls as the other plan routes. The route owns the plan, so body fields cannot escalate access.
- Research is a selectable perspective on Free and Integrated requests. Perspective does not grant paid export permissions.
- Add a Research & Academia persona, prose register and metadata entry, with methods, provenance, units, temporal/spatial coverage, comparability, uncertainty and identification limits. Do not invent publications, citations, significance tests or causal findings. Research is an audience lens, not a separate retrieval corpus.
- The paid Research plan allows supported cross-country work and the shared downloadable export capability, including CSV and GeoJSON. Reuse the export runner, storage, artifact schema and URL refresh flow.
- Add real GeoJSON generation to the shared export path. Validate provided geometries or WGS84 longitude/latitude points with an established library. Retain attribute-only observations as features with null geometry, disclose that limitation, and never geocode names or invent boundaries. Invalid coordinates fail explicitly. Preserve observation properties, units, sources and available citation IDs.
- Include the plan in tenant validation, rate limiting, model selection, API metadata, inspector choices, session defaults, OpenAPI and configuration/partner documentation. Do not alter production secrets or invent subscription prices or account entitlements.

## Acceptance

1. Authenticated Research and Integrated tenants can use `/query/research`; a Free tenant cannot. A Research tenant cannot select Integrated access by changing the route or body.
2. Both request enum fields accept `Research`. Free and Integrated requests can explicitly select that category while retaining their own plan permissions. Public labels use Research & Academia.
3. Prompt assembly includes the academic perspective, register and appropriate plan depth. Explicit category selection wins. Research has a natural plan perspective when no category is selected.
4. Research CSV, GeoJSON and combined CSV/GeoJSON requests produce validated downloadable artifact responses with the correct filename, MIME, data and provenance. Test points, existing polygons, missing geometry, invalid coordinates and nonstandard numeric/date properties.
5. Existing eligible plans can use the shared GeoJSON builder. Existing restricted plans still cannot export merely by selecting Research as their category.
6. The generated partner OpenAPI and public metadata agree with runtime routes, enums and artifact kinds. Existing route, export, session and persona tests continue to pass.

Deployment is outside this change. Actual paid accounts need their server-side tenant/subscription configuration to select `research`; adding a category is not authorization.

## Verification — 2026-11-20

- API, persona and export review gates passed after correcting deployment manifests, combined-format requests and strict geometry validation.
- Focused API, export, schema, profile and tenant regression suite: 141 passed. Final GeoJSON regressions: 18 passed. Isolated staged Research feature against committed source: 35 passed.
- Focused type checking: zero errors; Python compilation and staged whitespace checks passed.
- Full RAG and enterprise run: 3,742 passed, 24 failed, one skipped. All 22 deterministic failures also fail against the saved pre-Research sources; the other two pass individually on both versions. Broader RAG regression work remains open and this result is not a clean production release gate.
- Research accounts require server-side `plan_slug: "research"`; existing Integrated keys can test `/query/research`. Production artifact downloads require the existing S3/GCS configuration. GeoJSON retains observations without coordinates with null geometry.
