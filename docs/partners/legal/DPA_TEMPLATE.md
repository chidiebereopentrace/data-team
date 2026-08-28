# Data Processing Agreement (Template)

**Status:** Template for enterprise partners where OpenTrace processes personal or customer-confidential data. Not legal advice.

## 1. Roles

- **Controller:** Customer (e.g. Olam AGRI) determines purposes of processing end-user or employee queries
- **Processor:** OpenTrace processes query content, session metadata, and usage telemetry to deliver the Enterprise API

## 2. Subject matter and duration

Processing lasts for the MSA term plus retention periods in Section 6.

## 3. Nature and purpose of processing

- Natural-language agricultural intelligence queries
- Session continuity (session IDs, conversation summaries)
- API authentication and metering
- Optional observability (Langfuse traces when enabled)

## 4. Categories of data

- Query text and chat history supplied by Customer
- Optional `user_id` references supplied by Customer (pseudonymous identifiers recommended)
- API access logs (timestamps, tenant ID, token counts, IP addresses)

## 5. Processor obligations

OpenTrace shall:

- Process data only on documented Customer instructions
- Ensure personnel confidentiality
- Implement appropriate technical and organisational measures (encryption in transit, access controls, secret management)
- Assist with data subject requests where applicable
- Notify Customer of personal data breaches without undue delay
- Delete or return Customer Data on termination, subject to legal retention requirements

## 6. Sub-processors

Authorised sub-processors may include:

- Cloud infrastructure (GCP)
- LLM inference provider (OpenRouter or equivalent)
- Vector database (Qdrant Cloud)
- Redis session/cache provider
- Observability (Langfuse Cloud, when enabled)

Customer receives [30] days' notice of sub-processor changes.

## 7. International transfers

Transfers outside the Customer's jurisdiction use appropriate safeguards (SCCs or equivalent).

## 8. Audits

Customer may request a summary security questionnaire annually. On-site audits by mutual agreement with reasonable notice.

## 9. Contact

Data protection inquiries: contact@opentrace.africa
