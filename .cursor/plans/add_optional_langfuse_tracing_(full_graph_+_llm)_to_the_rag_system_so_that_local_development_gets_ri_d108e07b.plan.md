---
name: Add optional Langfuse tracing (full graph + LLM) to the RAG system so that local development gets rich observability while the HF Space deployment remains unchanged and tracing is gracefully disabled when keys are absent.
overview: ""
todos:
  - id: add-deps
    content: Add langfuse>=2.0 to ml-eng/requirements.txt and ml-eng/ml/rag/requirements.txt
    status: completed
  - id: env-docs
    content: Document LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in config/.env.example and ml-eng/ml/rag/README.md (HF deploy section)
    status: completed
  - id: create-obs-module
    content: Create ml-eng/ml/rag/observability.py that lazily initializes Langfuse client + LangfuseCallbackHandler only when keys are present (fail-open)
    status: completed
  - id: instrument-llm
    content: Wrap llm_chat_complete in ml-eng/ml/rag/llm_chat.py with @observe(as_type="generation") capturing model, usage, latency
    status: completed
  - id: instrument-graph
    content: Modify ml-eng/ml/rag/chatbot/graph.py:run_rag to accept/attach LangfuseCallbackHandler when available
    status: completed
  - id: instrument-api
    content: Update ml-eng/ml/rag/app/api.py:/query to start a root trace with session_id, stakeholder_type, query metadata and pass handler down
    status: completed
  - id: instrument-chat-turn
    content: Add spans in ml-eng/ml/rag/chat_turn.py:execute_chat_turn for session resolution and memory compaction
    status: completed
  - id: update-readme
    content: Add Langfuse section to ml-eng/ml/rag/README.md and ml-eng/deploy/README.md explaining local vs HF usage
    status: completed
  - id: hf-notes
    content: Confirm HF Space deploy instructions remain valid (no code changes needed in Dockerfile; tracing is opt-in via secrets)
    status: completed
  - id: test-plan
    content: "Local test: run Langfuse stack + RAG with keys, verify traces appear; then run without keys to confirm graceful fallback"
    status: in_progress
isProject: false
---

Add optional Langfuse tracing (full graph + LLM) to the RAG system so that local development gets rich observability while the HF Space deployment remains unchanged and tracing is gracefully disabled when keys are absent.