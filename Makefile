.PHONY: install test lint run-api run-streamlit run-mcp eval docker-build

VENV := venv/bin

install:
	python3 -m venv venv
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -r requirements.txt -r requirements-dev.txt
	$(VENV)/pip install -e .

test:
	$(VENV)/python3 -m pytest tests/ -v

lint:
	$(VENV)/python3 -m ruff check src/ tests/

run-api:
	$(VENV)/uvicorn serving.api:app --app-dir src --reload --port 8000

run-streamlit:
	$(VENV)/streamlit run src/app_oop.py

run-mcp:
	$(VENV)/python3 src/serving/mcp_server.py

eval:
	$(VENV)/python3 src/experiments/evaluate.py

docker-build:
	docker build -t financial-anomaly-detection-rag .
