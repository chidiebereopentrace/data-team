# Service Level Agreement (Template)

**Status:** Template for enterprise API partners. Attach to MSA as Exhibit A.

## 1. Covered services

Ask ADZA Enterprise Chatbot API v1 production endpoint(s) listed in the Order Form.

Excluded: sandbox environments, Customer-side integrations, third-party LLM or data provider outages beyond Provider's reasonable control.

## 2. Availability commitment

| Tier | Monthly uptime | Service credits |
|------|----------------|-----------------|
| Standard | 99.5% | 5% of monthly platform fee per 1% below target (max 25%) |
| Premium | 99.9% | 10% of monthly platform fee per 0.1% below target (max 50%) |

Uptime = `(total_minutes - downtime_minutes) / total_minutes` excluding scheduled maintenance.

## 3. Scheduled maintenance

- Up to [4] hours per month with [72] hours' notice
- Maintenance windows: [UTC window TBD, e.g. Sunday 02:00–06:00 UTC]

## 4. Support response times

| Priority | Definition | Initial response | Update cadence |
|----------|------------|------------------|----------------|
| P1 | Production API unavailable | [1] hour | Every [2] hours |
| P2 | Degraded performance or partial feature failure | [4] hours | Daily |
| P3 | Non-urgent questions, documentation | [1] business day | As needed |

Support channel: [dedicated email / Slack connect TBD]

## 5. Monitoring and status

Provider monitors `/v1/health` and backend readiness. Status page: [URL TBD].

## 6. Incident communication

Provider notifies Customer of P1 incidents within [1] hour of detection and provides a post-incident summary within [5] business days.

## 7. Service credits

Credits apply to the following month's invoice upon written request within [30] days of the incident month. Credits are the sole remedy for SLA breaches unless the MSA states otherwise.

## 8. Customer responsibilities

- Maintain valid API credentials and secure storage
- Report incidents with `request_id` from API responses
- Participate in reasonable root-cause analysis

**Contact:** contact@opentrace.africa
