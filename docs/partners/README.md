# Ask ADZA Enterprise API — Partner Documentation

Partner-facing materials for B2B integrations.

| Document | Audience |
|----------|----------|
| [ASKADZA_ENTERPRISE_API_OLAM_AGRI.docx](./ASKADZA_ENTERPRISE_API_OLAM_AGRI.docx) | Olam AGRI CEO — shareable Word brief |
| [ASKADZA_ENTERPRISE_API_OLAM_AGRI.md](./ASKADZA_ENTERPRISE_API_OLAM_AGRI.md) | Olam AGRI — pre-sales integration brief (source) |
| [ENTERPRISE_INTEGRATION_GUIDE.md](./ENTERPRISE_INTEGRATION_GUIDE.md) | Engineering teams — API integration |
| [openapi/chatbot-v1.json](./openapi/chatbot-v1.json) | OpenAPI 3 schema (regenerate via `ml-eng/scripts/export_chatbot_openapi.py`) |
| [legal/MSA_TEMPLATE.md](./legal/MSA_TEMPLATE.md) | Master Service Agreement template |
| [legal/DPA_TEMPLATE.md](./legal/DPA_TEMPLATE.md) | Data Processing Agreement template |
| [legal/SLA_TEMPLATE.md](./legal/SLA_TEMPLATE.md) | Service Level Agreement template |

Tenant provisioning: see `ml-eng/config/enterprise_tenants.example.json`.

Regenerate the Olam DOCX after editing the markdown brief:

```bash
cd ml-eng
PYTHONPATH=. python scripts/build_olam_enterprise_brief_docx.py
```
