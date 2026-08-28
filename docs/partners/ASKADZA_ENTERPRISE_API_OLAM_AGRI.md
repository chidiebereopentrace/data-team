# Ask ADZA Enterprise API
## Product Integration Brief for Olam AGRI

**Prepared by:** OpenTrace  
**Date:** August 2026  
**Classification:** Confidential — for Olam AGRI partnership discussions  
**Contact:** contact@opentrace.africa | [opentrace.africa](https://opentrace.africa) | [askadza.africa](https://askadza.africa)

---

## Executive summary

OpenTrace is Africa's agricultural intelligence layer — harmonising fragmented climate, production, market, and food-security data across 54 countries into decision-ready intelligence. **Ask ADZA** is the natural-language interface; the **Enterprise API** lets institutions like Olam AGRI embed that intelligence directly into procurement systems, risk dashboards, and internal tools — without building data pipelines or hiring analysts in the middle.

For Olam AGRI, the Enterprise API delivers:

- **Cross-country supply and production intelligence** for sourcing and portfolio decisions
- **Market volatility and price trend analysis** across African origins
- **Climate and yield risk signals** at national and sub-national levels
- **Traceable, confidence-scored answers** (ACF) with citations to underlying datasets
- **Exportable artifacts** (CSV, charts, DOCX, PDF) for reports and workflows

> *"Individuals use Ask ADZA; institutions connect via API."* — OpenTrace commercial model

---

## The problem Olam AGRI faces

Africa is central to global agri-supply chains, yet decision-makers often lack a unified view of:

- Production trends and yield stability across origins
- Climate variability and its impact on supply reliability
- Regional price dynamics and cross-border trade flows
- Emerging risks in specific districts and value chains

Data exists across governments, research bodies, and market systems — but it is **fragmented, inconsistent, and not queryable at decision speed**. OpenTrace closes that gap.

---

## What OpenTrace provides

OpenTrace is an **infrastructure company**, not a consultancy. Three pillars:

| Pillar | Role |
|--------|------|
| **OFIA** (OpenTrace Federated Intelligence Architecture) | Harmonises and version-controls datasets across global, national, and sub-national levels |
| **ACF** (ADZA Confidence Framework) | Attaches honest confidence signals to every answer based on evidence triangulation |
| **Ask ADZA** | Natural-language interface — web, mobile, WhatsApp, and **Enterprise API** |

**Coverage today:** 2B+ data points | 54 African countries | 12+ indicator domains (production, prices, trade, climate, food security, soil, employment, and more)

**Data domains relevant to Olam AGRI:**

- Crop production and yield trends (FAOSTAT, sub-national yield where available)
- Market prices and food balance sheets
- Cross-border trade flows
- Climate and vegetation indices (ERA5, NDVI, NASA POWER)
- Food security early warning (FEWS NET)
- Employment and macro indicators for market context

---

## Enterprise API — how it works

The Enterprise API is a **RESTful, versioned HTTP service** that accepts natural-language questions and returns structured, cited intelligence.

**Integration pattern:**

```
Olam application  →  POST /v1/chat/agribusinesses  →  OpenTrace intelligence engine
                     (API key in header)              →  Answer + citations + ACF + optional exports
```

**Authentication:** Send your API key on every request:

```
X-API-Key: <your-tenant-api-key>
```

Or:

```
Authorization: Bearer <your-tenant-api-key>
```

**Key capabilities for agribusiness partners:**

| Capability | Description |
|------------|-------------|
| Natural-language Q&A | Ask questions in plain English; no SQL or dashboards required |
| Multi-turn conversations | Server-side sessions or client-owned history for copilot-style UX |
| Cross-country comparison | Compare production, prices, or climate across African origins |
| Structured citations | Every claim links to source datasets (news, research, structured data) |
| ACF confidence scoring | Transparent `strong` / `moderate` / `limited` bands with explanation |
| Export artifacts | CSV, chart (PNG), DOCX, and PDF downloads via signed URLs |
| Usage metering | Token usage returned per request; monthly quotas enforced per tenant |

**Example questions Olam teams could ask via API:**

- *"Compare maize production trends in Nigeria, Ghana, and Côte d'Ivoire over the last five years."*
- *"Which regions in East Africa show the highest climate stress on coffee yields?"*
- *"What are retail maize price trends in Ethiopia and Kenya this season?"*
- *"Export a CSV of rice trade flows for West African countries in 2024."*

---

## Integration options for Olam AGRI

| Option | Description | Best for |
|--------|-------------|----------|
| **Embedded copilot** | API-backed chat widget in procurement or risk portals | Analysts and category managers |
| **Backend intelligence service** | Server-to-server calls from Olam's risk/scenario models | Automated monitoring and alerts |
| **Report generation** | Scheduled API queries → export artifacts → internal distribution | Weekly/monthly origin briefings |
| **WhatsApp / field channel** | Ask ADZA answers surfaced to field teams (future channel) | Origin managers and agronomists |

OpenTrace does **not** replace Olam's internal systems. Partners plug into shared intelligence infrastructure — data stays under original licences; OpenTrace monetizes **intelligence, not raw data**.

---

## Data trust and sovereignty

OpenTrace is built for institutional trust:

- **Partners retain ownership** of any data they contribute; attribution is preserved
- **Source data stays under its original licence** — OpenTrace does not relicense third-party datasets
- **No black-box scores** — every answer is interrogable via citations and ACF
- **Derived intelligence is separate** from source data
- **ACF triangulation** across global (25%), national/regional (40%), and ground/community (35%) evidence tiers

---

## Commercial model (indicative)

OpenTrace monetizes intelligence access, not data enclosure:

| Component | Model |
|-----------|-------|
| **Platform fee** | Annual enterprise licence for API access |
| **Usage** | Per-query or per-token consumption above included quota |
| **Exports** | Included in Agribusinesses/Integrated tier; volume tiers for high export use |
| **Custom scope** | Optional add-ons: additional countries, proprietary data integration, co-branded exports |
| **Pilot** | Reduced-fee or fee-waived sandbox period (typically 4–8 weeks) to validate use cases |

Specific pricing is scoped during partnership discovery based on query volume, countries, and integration depth.

---

## Technical requirements (Olam side)

Minimal integration footprint:

- **HTTPS client** capable of POST requests with JSON body
- **API key storage** in Olam's secrets management (server-side only; never in client apps)
- **Session handling** — reuse `session_id` for multi-turn, or pass `chat_history` for stateless calls
- **Citation rendering** — map inline `[N]` footnotes to citation cards in your UI (optional but recommended)
- **Export handling** — download artifacts from signed GCS URLs returned in `artifacts[]`

Full technical reference: [Enterprise Integration Guide](./ENTERPRISE_INTEGRATION_GUIDE.md) and [OpenAPI spec](./openapi/chatbot-v1.json).

---

## Proposed partnership journey

| Stage | Activities | Duration |
|-------|------------|----------|
| **1. Discovery** | Map Olam use cases (sourcing, risk, origins); agree success metrics | 1–2 weeks |
| **2. Sandbox** | API key, test environment, sample integration | 1 week |
| **3. Pilot** | Live integration in 2–3 workflows; joint evaluation of accuracy and citation quality | 4–8 weeks |
| **4. Production** | Production API, SLA, billing, partner success support | Ongoing |
| **5. Expand** | Additional use cases, optional proprietary data federation into OFIA | As needed |

**Pilot success metrics (examples):**

- ≥90% of pilot queries return actionable answers with citations
- ACF confidence distribution aligns with Olam analyst validation on sample set
- Time-to-insight reduced vs. manual data gathering baseline
- Export artifacts usable in existing Olam reporting workflows

---

## Why partner with OpenTrace

| For Olam AGRI | OpenTrace delivers |
|--------------|-------------------|
| Sourcing and supply risk | Cross-country production, climate, and market intelligence on demand |
| Speed to insight | Natural language instead of manual reconciliation across datasets |
| Institutional trust | ACF confidence scoring and full citation traceability |
| African coverage | 54 countries, continental scale, sub-national where data allows |
| Low integration burden | REST API — no data warehouse to build or maintain |
| Long-term infrastructure | OFIA persists and improves with use; not a one-off consultancy project |

OpenTrace already collaborates with agricultural research and development institutions (IITA, CGIAR ecosystem, AGRA, governments, and foundations). Olam AGRI would join a growing network of institutions leveraging shared agricultural intelligence infrastructure.

---

## Next steps

1. **Introductory call** — align on Olam priorities and candidate use cases
2. **NDA + sandbox provisioning** — API access for technical evaluation
3. **Discovery workshop** — Olam category/risk teams + OpenTrace product/engineering
4. **Pilot agreement** — scope, timeline, success criteria, and commercial terms
5. **Integration kickoff** — Olam engineering receives API credentials, OpenAPI spec, and integration guide

**Contact:** contact@opentrace.africa

---

## Appendix A — API surface summary (Agribusinesses tier)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/v1/health` | Service health |
| GET | `/v1/meta` | Plan types, categories, rate limits |
| GET | `/v1/usage` | Current tenant usage (authenticated) |
| POST | `/v1/sessions` | Create conversation session |
| GET | `/v1/sessions/{id}` | Session status |
| POST | `/v1/chat/agribusinesses` | **Primary endpoint** — Q&A with exports |
| POST | `/v1/feedback` | Optional quality feedback (when tracing enabled) |

**Request body (minimal):**

```json
{
  "message": "Compare cocoa production in Ghana and Côte d'Ivoire over the last 3 years",
  "session_id": "optional-for-multi-turn"
}
```

**Response includes:** `assistant_message`, `citations[]`, `acf`, `usage`, `artifacts[]` (when export requested), `session_id`, `request_id`

---

## Appendix B — OpenTrace readiness timeline

OpenTrace is completing the following for production B2B launch:

- API gateway and client authentication — **available in sandbox**
- Per-tenant metering and billing pipeline — **available in sandbox**
- Partner sandbox environment — **available on request**
- Published OpenAPI specification and integration guide — **available**
- Enterprise MSA, DPA, and SLA templates — **available under NDA**

Pilot can begin on a **controlled sandbox** while production gateway hardening completes in parallel.
