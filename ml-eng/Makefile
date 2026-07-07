.PHONY: help venv-install test rag-run rag-api rag-streamlit rag-streamlit-inspector

help:
	@echo "ML/AI-eng (ml-eng/) commands"
	@echo "  make venv-install   Install ml-eng deps into active venv"
	@echo "  make test           Run ML tests (excluding rag/serving by default)"
	@echo "  make rag-run        Run RAG CLI (uses Qdrant Cloud)"
	@echo "  make rag-api        Start FastAPI RAG server"
	@echo "  make rag-streamlit  Start Streamlit pipeline inspector"
	@echo "  make rag-streamlit-inspector  Alias for rag-streamlit"

venv-install:
	pip install -r ml-eng/requirements.txt -r ml-eng/requirements-dev.txt

test:
	PYTHONPATH=ml-eng pytest ml-eng/ml/ -v --ignore=ml-eng/ml/rag --ignore=ml-eng/ml/serving -x --tb=short

rag-run:
	PYTHONPATH=ml-eng python -m ml.rag.run

rag-api:
	PYTHONPATH=ml-eng uvicorn ml.rag.api:app --host 0.0.0.0 --port 7860

rag-streamlit:
	PYTHONPATH=ml-eng streamlit run ml-eng/ml/rag/chatbot/streamlit_app.py

rag-streamlit-inspector: rag-streamlit

