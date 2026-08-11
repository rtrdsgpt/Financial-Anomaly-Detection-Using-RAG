# Financial Anomaly Detection Using RAG

[![CI](https://github.com/rtrdsgpt/Financial-Anomaly-Detection-Using-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/rtrdsgpt/Financial-Anomaly-Detection-Using-RAG/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Detects statistically anomalous market moves (Z-score / Isolation Forest)
and explains *why*, using a real retrieval-augmented pipeline over
historical news: chunk -> embed -> vector-search -> rerank -> generate with
inline citations -> deterministically verify each citation against its
source text before trusting it.

Upgraded, independently-maintained fork of an MA5750 (Object-Oriented
Programming) course project -- see
[`rtrdsgpt/F.A.I.L._OOPS`](https://github.com/rtrdsgpt/F.A.I.L._OOPS) for
the original. This repo carries no shared git history with it.

## Table of Contents

- [Results](#results)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Scope](#scope)
- [License](#license)

## Results

No eval numbers are shown here on purpose -- `src/experiments/evaluate.py`
is real and runnable (dual metric: deterministic fact-overlap/citation-
coverage/embedding-similarity, plus an LLM-judge rubric), but this README
does not report scores that weren't actually produced by running it against
a live Groq API key. Run it yourself:

```bash
GROQ_API_KEY=... venv/bin/python3 src/experiments/evaluate.py
```

...and browse results in `results/eval/eval_report_*.json`, or
`venv/bin/mlflow ui` for the tracked runs.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full RAG
pipeline diagram and a walkthrough of the OOP design patterns in use
(Strategy, Factory, Builder, Template Method, Observer, Repository). Short
version:

```
src/
├── core/            interfaces.py, base.py, exceptions.py -- SOLID skeleton
├── processors/       Strategy classes: original pipeline (data_loader,
│                     anomaly_detector, news_retriever, embedding_generator,
│                     similarity_analyzer, ai_explainer) + RAG upgrade
│                     (chunking, vector_store, reranker, rag_retriever,
│                     grounded_explainer)
├── pipeline/         quantitative_pipeline.py (Template Method) ·
│                     pipeline_factory.py (Factory/Builder/Director) ·
│                     progress_observer.py (Observer)
├── ui/               Streamlit UI
├── config/           settings.py (pydantic-settings; env/.env, falls back
│                     to api_keys.txt if present)
├── experiments/       eval_set.json (13 synthetic fixtures) · metrics.py ·
│                     evaluate.py (dual-metric eval, logs to MLflow)
└── serving/          api.py (FastAPI) · schemas.py · dependencies.py ·
                      mcp_server.py (MCP tool server)
tests/                 pytest -- one file per Strategy family, offline/mocked
airflow/dags/          financial_anomaly_dag.py -- same Strategy classes, scheduled
.github/workflows/     CI: ruff + pytest + docker build
```

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt -r requirements-dev.txt
venv/bin/pip install -e .
cp .env.example .env   # fill in GROQ_API_KEY / FINNHUB_API_KEY
```

`FINNHUB_API_KEY` is optional (falls back to the free Yahoo Finance news
source); `GROQ_API_KEY` is required for any actual explanation generation
(`/explain`, `/analyze`, `evaluate.py`, the MCP `explain_anomaly` tool).

## Usage

**Run the full pipeline** (CLI):

```bash
venv/bin/python3 src/main_oop.py
```

**Streamlit UI:**

```bash
venv/bin/streamlit run src/app_oop.py
```

**Serve the API:**

```bash
venv/bin/uvicorn serving.api:app --app-dir src --reload --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/explain -H "Content-Type: application/json" -d '{
  "event": {"Date": "2025-01-15", "Ticker": "NVDA", "Event_Type": "Positive Outlier",
            "Z_score": 3.4, "Return": 0.09, "News_Headlines": "Beats EPS estimates by 18%"},
  "historical_context": []
}'
```

**As an MCP tool** (stdio -- add to Claude Desktop's config to call
`detect_anomalies`/`explain_anomaly` directly):

```bash
venv/bin/python3 src/serving/mcp_server.py
```

```json
{
  "mcpServers": {
    "financial-anomaly-detection-rag": {
      "command": "/absolute/path/to/venv/bin/python3",
      "args": ["/absolute/path/to/src/serving/mcp_server.py"]
    }
  }
}
```

**Scheduled orchestration (Airflow + Docker):**

```bash
docker compose up airflow-init   # first run only: migrates the DB, creates the admin user
docker compose up airflow-webserver airflow-scheduler postgres
```

Airflow UI at `http://localhost:8080` (admin/admin) -- the
`financial_anomaly_dag` DAG runs the same pipeline stages the CLI/API do,
configurable via Airflow Variables (`faildrag_ticker`, `faildrag_benchmark`,
etc., see `airflow/dags/financial_anomaly_dag.py`).

**Just the API, via Docker:**

```bash
docker compose up api
```

**Evaluate:**

```bash
venv/bin/python3 src/experiments/evaluate.py --help
```

## Testing

Offline/mocked -- no model downloads, no GPU, no network, no API keys:

```bash
venv/bin/python3 -m pytest tests/ -v
venv/bin/python3 -m ruff check src/ tests/
```

CI runs both on every push -- see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Scope

Deliberately bounded, on purpose:

- **Chroma (local, no server) or a pure-numpy fallback** as the vector
  store -- not Qdrant/pgvector; this project's retrieval corpus is one
  ticker's recent event history, not a scale that needs a hosted vector DB.
- **One reranker family** (cross-encoder via `sentence-transformers`, with
  a no-op fallback) -- not a hosted reranking API.
- **A custom dual-metric eval**, not `ragas` -- the deterministic half
  needed a citation-coverage metric tied directly to this project's own
  `CitationVerifier` output, which a generic RAG-eval library doesn't know
  about.
- **MLflow local file-store tracking + Airflow LocalExecutor** -- no
  hosted MLflow server, no Celery/Redis/worker fleet. One daily DAG doesn't
  need a worker fleet.
- **No fabricated eval numbers, ever** -- see [Results](#results).

## License

[MIT](LICENSE)
